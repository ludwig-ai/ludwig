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
import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import PREDICTIONS, TRAINER
from ludwig.utils.triton_utils import export_triton
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
)


def test_triton_torchscript(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # vector_feature(),
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
        # timeseries_feature(),
        # date_feature(),
        # h3_feature(),
        # set_feature(vocab_size=3),
        # bag_feature(vocab_size=3),
    ]
    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # set_feature(vocab_size=3),
        # vector_feature()
    ]
    backend = LocalTestBackend()
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    # Convert bool values to strings, e.g., {'Yes', 'No'}
    df = pd.read_csv(training_data_csv_path)
    df.to_csv(training_data_csv_path)

    # Train Ludwig (Pythonic) model:
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    # Obtain predictions from Python model
    preds_dict, _ = ludwig_model.predict(dataset=training_data_csv_path, return_type=dict)

    # Create graph inference model (Torchscript) from trained Ludwig model.
    triton_path = os.path.join(tmpdir, "triton")
    model_name = "test_triton"
    model_version = 1
    model_path, config_path = export_triton(ludwig_model, triton_path, model_name, model_version)

    # Validate relative path
    output_filename = os.path.relpath(model_path, triton_path)
    assert output_filename == f"{model_name}/{model_version}/model.pt"
    config_filename = os.path.relpath(config_path, triton_path)
    assert config_filename == f"{model_name}/config.pbtxt"

    # Restore the torchscript model
    restored_model = torch.jit.load(model_path)

    def to_input(s: pd.Series) -> Union[List[str], torch.Tensor]:
        if s.dtype == "object":
            return s.to_list()
        return torch.from_numpy(s.to_numpy().astype(np.float32))

    df = pd.read_csv(training_data_csv_path)
    inputs = {name: to_input(df[feature.column]) for name, feature in ludwig_model.model.input_features.items()}
    outputs = restored_model(**inputs)

    def from_output(o: Union[List[str], torch.Tensor]) -> np.array:
        if isinstance(o, list):
            return np.array(o)
        return o.numpy()

    # Enumerate over the output feature and lookup predictions to see the match outputs
    assert len(preds_dict) == len(outputs)
    for i, feature_name in enumerate(ludwig_model.model.output_features):
        output_values_expected = preds_dict[feature_name][PREDICTIONS]
        output_values = from_output(outputs[i])
        if output_values.dtype.type in {np.string_, np.str_}:
            # Strings should match exactly
            assert np.all(output_values == output_values_expected), f"feature: {feature_name}, output: predictions"
        else:
            assert np.allclose(output_values, output_values_expected), f"feature: {feature_name}, output: predictions"
