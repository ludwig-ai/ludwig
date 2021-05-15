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
import json
import logging
import os
import shutil
import sys

import pandas as pd
import pytest
from skimage.io import imread

from ludwig.api import LudwigModel
from ludwig.serve import server, ALL_FEATURES_PRESENT_ERROR
from ludwig.utils.data_utils import read_csv
from ludwig.utils.server_utils import serialize_payload
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import LocalTestBackend

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
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }
    model = LudwigModel(config, backend=LocalTestBackend())
    _, _, output_dir = model.train(
        dataset=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(dataset=data_csv, output_directory=output_dir)

    return model, output_dir


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


# todo clean up at end of PR 1174
# def convert_to_form(entry):
#     data = {}
#     files = []
#     for k, v in entry.items():
#         if type(v) == str and os.path.exists(v):
#             file = open(v, 'rb')
#             files.append((k, (v, file.read(), 'image/jpeg')))
#             file.close()
#         elif isinstance(v, np.ndarray):
#             # serialize ludwig specific format for numpy ndarray
#             data[k] = ndarray2string(v)
#         else:
#             data[k] = v
#     return data, files

# todo clean up at end of PR 1174
# def convert_to_batch_form(data_df):
#     # make copy so we don't affect source data
#     _data_df = data_df.copy()
#
#     # convert any ndarray columns to ludwig custom ndarray string format
#     for c in _data_df.columns:
#         if isinstance(_data_df[c][0], np.ndarray):
#             _data_df[c] = _data_df[c].apply(ndarray2string)
#
#     data = _data_df.to_dict(orient='split')
#     files = {
#         'dataset': (None, json.dumps(data), 'application/json'),
#     }
#     for row in data['data']:
#         for v in row:
#             if type(v) == str and os.path.exists(v) and v not in files:
#                 files[v] = (v, open(v, 'rb'), 'application/octet-stream')
#     return files


@pytest.mark.parametrize('img_source', ['file', 'ndarray'])
def test_server_integration(img_source, csv_filename):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
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
        category_feature(vocab_size=2),
        numerical_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    model, output_dir = train_model(input_features, output_features,
                                    data_csv=rel_path)

    app = server(model)
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200

    response = client.post('/predict')
    # expect the HTTP 400 error code for this situation
    assert response.status_code == 400
    assert response.json() == ALL_FEATURES_PRESENT_ERROR

    data_df = read_csv(rel_path)
    if img_source == 'ndarray':
        # convert image files to ndarrays into the dataframe
        image_feature_name = input_features[0]['name']
        data_df[image_feature_name] = data_df[image_feature_name].apply(
            lambda x: imread(x))

    # One-off prediction
    payload_dict, payload_files = serialize_payload(data_df.loc[0])
    if payload_files:
        server_response = client.post(
            '/predict',
            data={'payload': json.dumps(payload_dict)},
            files=payload_files
        )
    else:
        server_response = client.post(
            '/predict',
            data={'payload': json.dumps(payload_dict)}
        )

    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(list(server_response.keys()))
    assert server_response_keys == sorted(output_keys_for(output_features))

    model_output, _ = model.predict(
        dataset=pd.DataFrame(data_df.loc[0]).T
    )
    model_output = model_output.to_dict('records')[0]
    assert model_output == server_response

    # Batch prediction
    assert len(data_df) > 1
    payload_dict, payload_files = serialize_payload(data_df)
    if payload_files:
        server_response = client.post(
            '/batch_predict',
            data={'payload': json.dumps(payload_dict)},
            files=payload_files
        )
    else:
        server_response = client.post(
            '/batch_predict',
            data={'payload': json.dumps(payload_dict)}
        )
    assert server_response.status_code == 200
    server_response = server_response.json()

    server_response_keys = sorted(server_response['columns'])
    assert server_response_keys == sorted(output_keys_for(output_features))
    assert len(data_df) == len(server_response['data'])

    model_output, _ = model.predict(dataset=data_df)
    model_output = model_output.to_dict('split')
    assert model_output == server_response

    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(image_dest_folder, ignore_errors=True)
