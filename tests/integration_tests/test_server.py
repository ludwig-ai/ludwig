# -*- coding: utf-8 -*-
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
import logging
import os
import shutil
import sys

from ludwig.api import LudwigModel
from ludwig.serve import server, ALL_FEATURES_PRESENT_ERROR
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import text_feature

logger = logging.getLogger(__name__)

try:
    from starlette.testclient import TestClient
except ImportError:
    logger.error(
        ' fastapi and other serving dependencies are not installed. '
        'In order to install all serving dependencies run '
        'pip install ludwig[serve]'
    )
    sys.exit(-1)


def train_model(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)

    # Training with csv
    model.train(
        data_csv=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )

    model.predict(data_csv=data_csv)

    # Remove results/intermediate data saved to disk
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)

    # Training with dataframe
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_df=data_df)
    return model


def output_keys_for(output_features):
    keys = []
    for feature in output_features:
        name = feature['name']
        if feature['type'] == 'category':
            keys.append("{}_predictions".format(name))
            keys.append("{}_probability".format(name))
            keys.append("{}_probabilities_<UNK>".format(name))
            for category in feature['idx2str']:
                keys.append(
                    "{}_probabilities_{}".format(name, category))

        elif feature['type'] == 'numerical':
            keys.append("{}_predictions".format(name))
        else:
            raise NotImplementedError
    return keys


def convert_to_form(entry):
    data = {}
    files = []
    for k, v in entry.items():
        if type(v) == str and os.path.exists(v):
            file = open(v, 'rb')
            files.append((k, (v, file.read(), 'image/jpeg')))
            file.close()
        else:
            data[k] = v
    return data, files


def test_server_integration(csv_filename):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3
            },
            fc_size=16,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1),
        numerical_feature(normalization='zscore')
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        numerical_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    model = train_model(input_features, output_features, data_csv=rel_path)

    app = server(model)
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200

    response = client.post('/predict')
    assert response.json() == ALL_FEATURES_PRESENT_ERROR

    data_df = read_csv(rel_path)
    data, files = convert_to_form(data_df.T.to_dict()[0])
    response = client.post('/predict', data=data, files=files)

    response_keys = sorted(list(response.json().keys()))
    assert response_keys == sorted(output_keys_for(output_features))

    shutil.rmtree(model.exp_dir_name, ignore_errors=True)
    shutil.rmtree(image_dest_folder)
