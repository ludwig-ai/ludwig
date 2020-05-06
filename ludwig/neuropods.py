import os
import shutil

from ludwig.api import LudwigModel
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, \
    TRAIN_SET_METADATA_FILE_NAME, MODEL_WEIGHTS_FILE_NAME
from ludwig.utils.data_utils import load_json


class LudwigNeuropodModelWrapper:
    def __init__(self, data_root):
        self.ludwig_model = LudwigModel.load(data_root)

    def __call__(self, **kwargs):
        print('__call__', file=sys.stderr)
        return self.ludwig_model.predict(data_dict=kwargs, return_type=dict)


def get_model(data_root):
    print('get_model()', data_root, file=sys.stderr)
    return LudwigNeuropodModelWrapper(data_root)


def build_neuropod(
        ludwig_model_path,
        neuropod_path="neuropod"
):
    from neuropods.backends.python.packager import create_python_neuropod

    data_paths = [
        {
            "path": ludwig_model_path,
            "packaged_name": MODEL_HYPERPARAMETERS_FILE_NAME
        },
        {
            "path": ludwig_model_path,
            "packaged_name": MODEL_WEIGHTS_FILE_NAME + ".meta"
        },
        {
            "path": ludwig_model_path,
            "packaged_name": MODEL_WEIGHTS_FILE_NAME + ".index"
        },
        {
            "path": ludwig_model_path,
            "packaged_name": TRAIN_SET_METADATA_FILE_NAME
        },
    ]
    model_weights_files_pattern = MODEL_WEIGHTS_FILE_NAME + ".data-"
    for filename in os.listdir(ludwig_model_path):
        if model_weights_files_pattern in filename:
            data_paths.append(
                {
                    "path": ludwig_model_path,
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
            "dtype": "string",
            "shape": ("batch_size", 1)
        })
    output_spec = []
    for feature in ludwig_model_definition['output_features']:
        output_spec.append({
            "name": feature['name'],
            "dtype": "string",
            "shape": ("batch_size", 1)
        })

    shutil.rmtree(neuropod_path, ignore_errors=True)

    create_python_neuropod(
        neuropod_path=neuropod_path,
        model_name="ludwig_model",
        data_paths=data_paths,
        code_path_spec=[{
            "python_root": 'code',
            "dirs_to_package": [
                ""  # Package everything in the python_root
            ],
        }],
        entrypoint_package="main",
        entrypoint="get_model",
        # test_deps=['torch', 'numpy'],
        skip_virtualenv=True,
        input_spec=input_spec,
        output_spec=output_spec
    )
