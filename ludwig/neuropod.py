import os
import shutil
import sys

import numpy as np

from ludwig.api import LudwigModel
from ludwig.constants import CATEGORY, NUMERICAL, BINARY, SEQUENCE, TEXT, SET
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, \
    TRAIN_SET_METADATA_FILE_NAME, MODEL_WEIGHTS_FILE_NAME
from ludwig.utils.data_utils import load_json


class LudwigNeuropodModelWrapper:
    def __init__(self, data_root):
        self.ludwig_model = LudwigModel.load(data_root)

    def __call__(self, **kwargs):
        print('__call__', file=sys.stderr)
        predicted = self.ludwig_model.predict(
            data_dict=kwargs, return_type=dict
        )
        print(predicted, file=sys.stderr)
        return postprocess_for_neuropod(
            predicted, self.ludwig_model.model_definition
        )


def get_model(data_root):
    print('get_model()', data_root, file=sys.stderr)
    return LudwigNeuropodModelWrapper(data_root)


def postprocess_for_neuropod(predicted, model_definition):
    postprocessed = {}
    for output_feature in model_definition['output_features']:
        feature_name = output_feature['name']
        feature_type = output_feature['type']
        if feature_type == BINARY:
            postprocessed[feature_name] = np.array(
                predicted[feature_name]['_predictions'], dtype='bool'
            )
            postprocessed[feature_name + "_probability"] = \
                predicted[feature_name]['probability']
        elif feature_type == NUMERICAL:
            postprocessed[feature_name + "_predictions"] = \
                predicted[feature_name]['predictions']
        elif feature_type == CATEGORY:
            postprocessed[feature_name + "_predictions"] = np.array(
                predicted[feature_name]['predictions'], dtype='str'
            )
            postprocessed[feature_name + "_probability"] = \
                predicted[feature_name]['probability']
            postprocessed[feature_name + "_probabilities"] = \
                predicted[feature_name]['probabilities']
        elif feature_type == SEQUENCE:
            postprocessed[feature_name + "_predictions"] = np.array(
                predicted[feature_name]['predictions'], dtype='str'
            )
        elif feature_type == TEXT:
            postprocessed[feature_name + "_predictions"] = np.array(
                predicted[feature_name]['predictions'], dtype='str'
            )
        elif feature_type == SET:
            postprocessed[feature_name + "_predictions"] = np.array(
                predicted[feature_name]['predictions'], dtype='str'
            )
            postprocessed[feature_name + "_probability"] = \
                predicted[feature_name]['probability']
            postprocessed[feature_name + "_probabilities"] = \
                predicted[feature_name]['probabilities']
        else:
            postprocessed[feature_name + "_predictions"] = np.array(
                predicted[feature_name]['predictions'], dtype='str'
            )
    print(postprocessed, file=sys.stderr)
    return postprocessed


def build_neuropod(
        ludwig_model_path,
        neuropod_path="/Users/piero/Desktop/neuropod",
        python_root="/Users/piero/Development/ludwig"
):
    from neuropod.backends.python.packager import create_python_neuropod

    data_paths = [
        {
            "path": os.path.join(
                ludwig_model_path, MODEL_HYPERPARAMETERS_FILE_NAME
            ),
            "packaged_name": MODEL_HYPERPARAMETERS_FILE_NAME
        },
        {
            "path": os.path.join(
                ludwig_model_path, TRAIN_SET_METADATA_FILE_NAME
            ),
            "packaged_name": TRAIN_SET_METADATA_FILE_NAME
        },
        {
            "path": os.path.join(
                ludwig_model_path, 'checkpoint'
            ),
            "packaged_name": 'checkpoint'
        },
    ]
    for filename in os.listdir(ludwig_model_path):
        if MODEL_WEIGHTS_FILE_NAME in filename:
            data_paths.append(
                {
                    "path": os.path.join(
                        ludwig_model_path, filename
                    ),
                    "packaged_name": filename
                }
            )

    ludwig_model_definition = load_json(
        os.path.join(
            ludwig_model_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
    )
    input_spec = []
    for feature in ludwig_model_definition['input_features']:
        input_spec.append({
            "name": feature['name'],
            "dtype": "str",
            "shape": (None,)
        })
    output_spec = []
    for feature in ludwig_model_definition['output_features']:
        feature_type = feature['type']
        if feature_type == BINARY:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "bool",
                "shape": (None,)
            })
            output_spec.append({
                "name": feature['name'] + '_probability',
                "dtype": "float32",
                "shape": (None,)
            })
        elif feature_type == NUMERICAL:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "float32",
                "shape": (None,)
            })
        elif feature_type == CATEGORY:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "str",
                "shape": (None,)
            })
            output_spec.append({
                "name": feature['name'] + '_probability',
                "dtype": "float32",
                "shape": (None,)
            })
            output_spec.append({
                "name": feature['name'] + '_probabilities',
                "dtype": "float32",
                "shape": (None, None)
            })
        elif feature_type == SEQUENCE:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "str",
                "shape": (None,)
            })
        elif feature_type == TEXT:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "str",
                "shape": (None,)
            })
        elif feature_type == SET:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "str",
                "shape": (None,)
            })
            output_spec.append({
                "name": feature['name'] + '_probability',
                "dtype": "float32",
                "shape": (None,)
            })
            output_spec.append({
                "name": feature['name'] + '_probabilities',
                "dtype": "float32",
                "shape": (None, None)
            })
        else:
            output_spec.append({
                "name": feature['name'] + '_predictions',
                "dtype": "str",
                "shape": (None,)
            })

    if os.path.exists(neuropod_path):
        if os.path.isfile(neuropod_path):
            os.remove(neuropod_path)
        else:
            shutil.rmtree(neuropod_path, ignore_errors=True)

    create_python_neuropod(
        neuropod_path=neuropod_path,
        model_name="ludwig_model",
        data_paths=data_paths,
        code_path_spec=[{
            "python_root": python_root,
            "dirs_to_package": [
                "ludwig"  # Package everything in the python_root
            ],
        }],
        entrypoint_package="ludwig.neuropod",
        entrypoint="get_model",
        # test_deps=['torch', 'numpy'],
        skip_virtualenv=True,
        input_spec=input_spec,
        output_spec=output_spec
    )
