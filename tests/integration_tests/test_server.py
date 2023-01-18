# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import logging
import os
import sys

import numpy as np
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, DECODER, TRAINER
from ludwig.serve import server
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import (
    audio_feature,
    category_feature,
    generate_data,
    image_feature,
    LocalTestBackend,
    number_feature,
    text_feature,
)

logger = logging.getLogger(__name__)

ALL_FEATURES_PRESENT_ERROR = "Data received does not contain all input features"

try:
    from starlette.testclient import TestClient
except ImportError:
    logger.error(
        " fastapi and other serving dependencies are not installed. "
        "In order to install all serving dependencies run "
        "pip install ludwig[serve]"
    )
    sys.exit(-1)


def train_and_predict_model(input_features, output_features, data_csv, output_directory):
    """Helper method to avoid code repetition for training a model and using it for prediction.

    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :param output_directory: model output directory
    :return: None
    """
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    model = LudwigModel(config, backend=LocalTestBackend())
    model.train(
        dataset=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        output_directory=output_directory,
    )
    model.predict(dataset=data_csv, output_directory=output_directory)
    return model


def train_and_predict_model_with_stratified_split(input_features, output_features, data_csv, output_directory):
    """Same as above, but with stratified split."""
    print(f'output_features[0]["column"]: {output_features[0]["column"]}')
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
        "preprocessing": {
            "split": {"column": output_features[0]["column"], "probabilities": [0.7, 0.1, 0.2], "type": "stratify"},
        },
    }
    model = LudwigModel(config, backend=LocalTestBackend())
    model.train(
        dataset=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        output_directory=output_directory,
    )
    model.predict(dataset=data_csv, output_directory=output_directory)
    return model


def output_keys_for(output_features):
    keys = []
    for feature in output_features:
        name = feature["name"]
        if feature["type"] == "category":
            keys.append(f"{name}_predictions")
            keys.append(f"{name}_probability")
            keys.append(f"{name}_probabilities")
            for category in feature[DECODER]["idx2str"]:
                keys.append(f"{name}_probabilities_{category}")

        elif feature["type"] == "number":
            keys.append(f"{name}_predictions")
        else:
            raise NotImplementedError
    return keys


def convert_to_form(entry):
    data = {}
    files = []
    for k, v in entry.items():
        if type(v) == str and os.path.exists(v):
            file = open(v, "rb")
            files.append((k, (v, file.read(), "application/octet-stream")))
        else:
            data[k] = v
    return data, files


def convert_to_batch_form(data_df):
    data = data_df.to_dict(orient="split")
    files = {
        "dataset": (None, json.dumps(data), "application/json"),
    }
    for row in data["data"]:
        for v in row:
            if type(v) == str and os.path.exists(v) and v not in files:
                files[v] = (v, open(v, "rb"), "application/octet-stream")
    return files


def test_server_integration_with_images(tmpdir):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder={"output_size": 16, "num_filters": 8},
            preprocessing={"in_memory": True, "height": 32, "width": 32, "num_channels": 3},
        ),
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="zscore"),
    ]
    output_features = [category_feature(decoder={"vocab_size": 4}), number_feature()]

    np.random.seed(123)  # reproducible synthetic data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    model = train_and_predict_model(input_features, output_features, data_csv=rel_path, output_directory=tmpdir)

    app = server(model)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

    response = client.post("/predict")
    # expect the HTTP 400 error code for this situation
    assert response.status_code == 400
    assert ALL_FEATURES_PRESENT_ERROR in str(response.json())

    data_df = read_csv(rel_path)

    # One-off prediction
    first_entry = data_df.T.to_dict()[0]
    data, files = convert_to_form(first_entry)
    server_response = client.post("/predict", data=data, files=files)
    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(list(server_response.keys()))
    assert server_response_keys == sorted(output_keys_for(output_features))

    model_output, _ = model.predict(dataset=[first_entry], data_format=dict)
    model_output = model_output.to_dict("records")[0]
    assert model_output == server_response

    # Batch prediction
    assert len(data_df) > 1
    files = convert_to_batch_form(data_df)
    server_response = client.post("/batch_predict", files=files)
    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(server_response["columns"])
    assert server_response_keys == sorted(output_keys_for(output_features))
    assert len(data_df) == len(server_response["data"])

    model_output, _ = model.predict(dataset=data_df)
    model_output = model_output.to_dict("split")
    assert model_output == server_response


def test_server_integration_with_stratified_split(tmpdir):
    input_features = [
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="zscore"),
    ]
    output_features = [category_feature(decoder={"vocab_size": 4})]

    np.random.seed(123)  # reproducible synthetic data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100)

    model = train_and_predict_model_with_stratified_split(
        input_features, output_features, data_csv=rel_path, output_directory=tmpdir
    )

    app = server(model)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

    response = client.post("/predict")
    # expect the HTTP 400 error code for this situation
    assert response.status_code == 400
    assert ALL_FEATURES_PRESENT_ERROR in str(response.json())

    data_df = read_csv(rel_path)

    # One-off prediction
    first_entry = data_df.T.to_dict()[0]
    data, files = convert_to_form(first_entry)
    server_response = client.post("/predict", data=data, files=files)
    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(list(server_response.keys()))
    assert server_response_keys == sorted(output_keys_for(output_features))

    model_output, _ = model.predict(dataset=[first_entry], data_format=dict)
    model_output = model_output.to_dict("records")[0]
    assert model_output == server_response

    # Batch prediction
    assert len(data_df) > 1
    files = convert_to_batch_form(data_df)
    server_response = client.post("/batch_predict", files=files)
    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(server_response["columns"])
    assert server_response_keys == sorted(output_keys_for(output_features))
    assert len(data_df) == len(server_response["data"])

    model_output, _ = model.predict(dataset=data_df)
    model_output = model_output.to_dict("split")
    assert model_output == server_response


@pytest.mark.parametrize("single_record", [False, True])
def test_server_integration_with_audio(single_record, tmpdir):
    # Audio Inputs
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    # Resnet encoder
    input_features = [
        audio_feature(
            folder=audio_dest_folder,
        ),
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="zscore"),
    ]
    output_features = [category_feature(decoder={"vocab_size": 4}), number_feature()]

    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    model = train_and_predict_model(input_features, output_features, data_csv=rel_path, output_directory=tmpdir)

    app = server(model)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

    response = client.post("/predict")
    # expect the HTTP 400 error code for this situation
    assert response.status_code == 400
    assert ALL_FEATURES_PRESENT_ERROR in str(response.json())

    data_df = read_csv(rel_path)

    if single_record:
        # Single record prediction
        first_entry = data_df.T.to_dict()[0]
        data, files = convert_to_form(first_entry)
        server_response = client.post("/predict", data=data, files=files)
        assert server_response.status_code == 200
        server_response = server_response.json()

        server_response_keys = sorted(list(server_response.keys()))
        assert server_response_keys == sorted(output_keys_for(output_features))

        model_output, _ = model.predict(dataset=[first_entry], data_format=dict)
        model_output = model_output.to_dict("records")[0]
        assert model_output == server_response
    else:
        # Batch prediction
        assert len(data_df) > 1
        files = convert_to_batch_form(data_df)
        server_response = client.post("/batch_predict", files=files)
        assert server_response.status_code == 200
        server_response = server_response.json()

        server_response_keys = sorted(server_response["columns"])
        assert server_response_keys == sorted(output_keys_for(output_features))
        assert len(data_df) == len(server_response["data"])

        model_output, _ = model.predict(dataset=data_df)
        model_output = model_output.to_dict("split")
        assert model_output == server_response
