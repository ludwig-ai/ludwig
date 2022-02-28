import importlib.util
import logging
import os
import shutil
import tempfile
from typing import Any, Dict

import numpy as np
import torch

from ludwig import __file__ as ludwig_path
from ludwig.api import LudwigModel
from ludwig.backend import LOCAL_BACKEND, LocalBackend
from ludwig.constants import (
    BINARY,
    CATEGORY,
    NAME,
    NUMBER,
    PREDICTIONS,
    PROBABILITIES,
    PROBABILITY,
    SEQUENCE,
    SET,
    TEXT,
    TYPE,
    VECTOR,
)
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, MODEL_WEIGHTS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils.data_utils import load_json

logger = logging.getLogger(__name__)


INFERENCE_MODULE_TEMPLATE = """
import torch
from typing import Any, Dict, List, Union

class GeneratedInferenceModule(torch.nn.Module):
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module

    def forward(self, {input_signature}):
        inputs = {input_dict}
        results = self.inference_module(inputs)
        #postprocess_for_neuropod(predicted, self.ludwig_model.config)
        return results
"""


def _get_input_signature(config: Dict[str, Any]) -> str:
    args = []
    for feature in config["input_features"]:
        name = feature[NAME]
        args.append(f"{name}: Union[List[str], torch.Tensor]")
    return ", ".join(args)


def _get_input_dict(config: Dict[str, Any]) -> str:
    elems = []
    for feature in config["input_features"]:
        name = feature[NAME]
        elems.append(f'"{name}": {name}')
    return "{" + ", ".join(elems) + "}"


def generate_neuropod_torchscript(model: LudwigModel):
    config = model.config
    inference_module = model.to_torchscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "generated.py")
        with open(ts_path, "w") as f:
            f.write(
                INFERENCE_MODULE_TEMPLATE.format(
                    input_signature=_get_input_signature(config), input_dict=_get_input_dict(config)
                )
            )

        spec = importlib.util.spec_from_file_location("generated.ts", ts_path)
        gen_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_ts)

        gen_module = gen_ts.GeneratedInferenceModule(inference_module)
        scripted_module = torch.jit.script(gen_module)
    return scripted_module


class LudwigNeuropodModelWrapper:
    def __init__(self, data_root, backend):
        self.ludwig_model = LudwigModel.load(data_root, backend=backend)

    def __call__(self, **kwargs):
        data_dict = kwargs
        for key in data_dict:
            data_dict[key] = np.squeeze(data_dict[key], axis=1)
        predicted, _ = self.ludwig_model.predict(dataset=data_dict, return_type="dict")
        # print(predicted, file=sys.stderr)
        return postprocess_for_neuropod(predicted, self.ludwig_model.config)


def get_model(data_root):
    return LudwigNeuropodModelWrapper(data_root, backend=LOCAL_BACKEND)


def get_test_model(data_root):
    class LocalTestBackend(LocalBackend):
        @property
        def supports_multiprocessing(self):
            return False

    return LudwigNeuropodModelWrapper(data_root, backend=LocalTestBackend())


def postprocess_for_neuropod(predicted, config):
    postprocessed = {}
    for output_feature in config["output_features"]:
        feature_name = output_feature[NAME]
        feature_type = output_feature[TYPE]
        if feature_type == BINARY:
            postprocessed[feature_name + "_predictions"] = np.expand_dims(
                predicted[feature_name][PREDICTIONS].astype("str"), 1
            )
            postprocessed[feature_name + "_probability"] = np.expand_dims(
                predicted[feature_name][PROBABILITY].astype("float64"), 1
            )
            postprocessed[feature_name + "_probabilities"] = predicted[feature_name][PROBABILITIES].astype("float64")
        elif feature_type == NUMBER:
            postprocessed[feature_name + "_predictions"] = np.expand_dims(
                predicted[feature_name][PREDICTIONS].astype("float64"), 1
            )
        elif feature_type == CATEGORY:
            postprocessed[feature_name + "_predictions"] = np.expand_dims(
                np.array(predicted[feature_name][PREDICTIONS], dtype="str"), 1
            )
            postprocessed[feature_name + "_probability"] = np.expand_dims(
                predicted[feature_name][PROBABILITY].astype("float64"), 1
            )
            postprocessed[feature_name + "_probabilities"] = predicted[feature_name][PROBABILITIES].astype("float64")
        elif feature_type == SEQUENCE or feature_type == TEXT:
            predictions = list(map(lambda x: " ".join(x), predicted[feature_name][PREDICTIONS]))
            postprocessed[feature_name + "_predictions"] = np.expand_dims(np.array(predictions, dtype="str"), 1)
        elif feature_type == SET:
            predictions = list(map(lambda x: " ".join(x), predicted[feature_name][PREDICTIONS]))
            postprocessed[feature_name + "_predictions"] = np.expand_dims(np.array(predictions, dtype="str"), 1)
            probability = list(map(lambda x: " ".join([str(e) for e in x]), predicted[feature_name]["probability"]))
            postprocessed[feature_name + "_probability"] = np.expand_dims(np.array(probability, dtype="str"), 1)
            postprocessed[feature_name + "_probabilities"] = predicted[feature_name][PROBABILITIES].astype("float64")
        elif feature_type == VECTOR:
            postprocessed[feature_name + "_predictions"] = predicted[feature_name][PREDICTIONS].astype("float64")
        else:
            postprocessed[feature_name + "_predictions"] = np.expand_dims(
                np.array(predicted[feature_name][PREDICTIONS], dtype="str"), 1
            )
    return postprocessed


def export_neuropod(
    ludwig_model_path,
    neuropod_path,
    neuropod_model_name="ludwig_model",
    entrypoint="get_model",
):
    try:
        from neuropod.backends.python.packager import create_python_neuropod
    except ImportError:
        raise RuntimeError('The "neuropod" package is not installed in your environment.')

    data_paths = [
        {
            "path": os.path.join(ludwig_model_path, MODEL_HYPERPARAMETERS_FILE_NAME),
            "packaged_name": MODEL_HYPERPARAMETERS_FILE_NAME,
        },
        {
            "path": os.path.join(ludwig_model_path, TRAIN_SET_METADATA_FILE_NAME),
            "packaged_name": TRAIN_SET_METADATA_FILE_NAME,
        },
        {"path": os.path.join(ludwig_model_path, "checkpoint"), "packaged_name": "checkpoint"},
    ]
    for filename in os.listdir(ludwig_model_path):
        if MODEL_WEIGHTS_FILE_NAME in filename:
            data_paths.append({"path": os.path.join(ludwig_model_path, filename), "packaged_name": filename})

    logger.debug(f"data_paths: {data_paths}")

    ludwig_config = load_json(os.path.join(ludwig_model_path, MODEL_HYPERPARAMETERS_FILE_NAME))
    training_set_metadata = load_json(os.path.join(ludwig_model_path, TRAIN_SET_METADATA_FILE_NAME))

    input_spec = []
    for feature in ludwig_config["input_features"]:
        input_spec.append({NAME: feature[NAME], "dtype": "str", "shape": (None, 1)})
    logger.debug(f"input_spec: {input_spec}")

    output_spec = []
    for feature in ludwig_config["output_features"]:
        feature_type = feature[TYPE]
        feature_name = feature[NAME]
        if feature_type == BINARY:
            output_spec.append({"name": feature_name + "_predictions", "dtype": "str", "shape": (None, 1)})
            output_spec.append({"name": feature_name + "_probability", "dtype": "float64", "shape": (None, 1)})
            output_spec.append({"name": feature_name + "_probabilities", "dtype": "float64", "shape": (None, 2)})
        elif feature_type == NUMBER:
            output_spec.append({"name": feature_name + "_predictions", "dtype": "float64", "shape": (None, 1)})
        elif feature_type == CATEGORY:
            output_spec.append({"name": feature_name + "_predictions", "dtype": "str", "shape": (None, 1)})
            output_spec.append({"name": feature_name + "_probability", "dtype": "float64", "shape": (None, 1)})
            output_spec.append(
                {
                    "name": feature_name + "_probabilities",
                    "dtype": "float64",
                    "shape": (None, training_set_metadata[feature[NAME]]["vocab_size"]),
                }
            )
        elif feature_type == SEQUENCE or feature_type == TEXT:
            output_spec.append({"name": feature_name + "_predictions", "dtype": "str", "shape": (None, 1)})
        elif feature_type == SET:
            output_spec.append({"name": feature_name + "_predictions", "dtype": "str", "shape": (None, 1)})
            output_spec.append({"name": feature_name + "_probability", "dtype": "str", "shape": (None, 1)})
            output_spec.append(
                {
                    "name": feature_name + "_probabilities",
                    "dtype": "float64",
                    "shape": (None, training_set_metadata[feature[NAME]]["vocab_size"]),
                }
            )
        elif feature_type == VECTOR:
            output_spec.append(
                {
                    "name": feature["name"] + "_predictions",
                    "dtype": "float64",
                    "shape": (None, training_set_metadata[feature[NAME]]["vector_size"]),
                }
            )
        else:
            output_spec.append({"name": feature["name"] + "_predictions", "dtype": "str", "shape": (None, 1)})
    logger.debug(f"output_spec: {output_spec}")

    if os.path.exists(neuropod_path):
        if os.path.isfile(neuropod_path):
            logger.warning(f"Removing file: {neuropod_path}")
            os.remove(neuropod_path)
        else:
            logger.warning(f"Removing directory: {neuropod_path}")
            shutil.rmtree(neuropod_path, ignore_errors=True)

    from pathlib import Path

    path = Path(ludwig_path)
    logger.debug(f"python_root: {path.parent.parent}")

    create_python_neuropod(
        neuropod_path=neuropod_path,
        model_name=neuropod_model_name,
        data_paths=data_paths,
        code_path_spec=[
            {
                "python_root": path.parent.parent,
                "dirs_to_package": ["ludwig"],  # Package everything in the python_root
            }
        ],
        entrypoint_package="ludwig.utils.neuropod_utils",
        entrypoint=entrypoint,
        skip_virtualenv=True,
        input_spec=input_spec,
        output_spec=output_spec,
    )
    logger.info(f"Neuropod saved to: {neuropod_path}")
