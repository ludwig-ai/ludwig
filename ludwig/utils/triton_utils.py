import importlib.util
import os
import tempfile
from typing import Any, Dict, Union

import torch

from ludwig.api import LudwigModel
from ludwig.constants import NAME

INFERENCE_MODULE_TEMPLATE = """
from typing import Any, Dict, List, Union
import torch
from ludwig.utils.types import TorchscriptPreprocessingInput

class GeneratedInferenceModule(torch.nn.Module):
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module

    def forward(self, {input_signature}):
        inputs: Dict[str, TorchscriptPreprocessingInput] = {input_dict}
        results = self.inference_module(inputs)
        return {output_tuple}
"""

TRITON_SPEC = """
    {{
        name: "{prefix}__{index}" # "{name}"
        data_type: {data_type}
        dims: [ {data_dims} ]
    }}"""

INSTANCE_SPEC = """
    {{
        count: {count}
        kind: {kind}
    }}"""

TRITON_CONFIG_TEMPLATE = """
name: "pytorch_raw"
platform: "pytorch_libtorch"
max_batch_size: 0 # Disable dynamic batching?
input [{input_spec}
]
output [{output_spec}
]
instance_group [{instance_spec}
]
"""


def _get_input_signature(config: Dict[str, Any]) -> str:
    args = []
    for feature in config["input_features"]:
        name = feature[NAME]
        args.append(f"{name}: TorchscriptPreprocessingInput")
    return ", ".join(args)


def _get_input_dict(config: Dict[str, Any]) -> str:
    elems = []
    for feature in config["input_features"]:
        name = feature[NAME]
        elems.append(f'"{name}": {name}')
    return "{" + ", ".join(elems) + "}"


def _get_output_tuple(config: Dict[str, Any]) -> str:
    results = []
    for feature in config["output_features"]:
        name = feature[NAME]
        results.append(f'results["{name}"]["predictions"]')
    return "(" + ", ".join(results) + ")"


def generate_triton_torchscript(model: LudwigModel) -> torch.jit.ScriptModule:
    """Generates a torchscript model in the triton format."""
    config = model.config
    inference_module = model.to_torchscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "generated.py")
        with open(ts_path, "w") as f:
            f.write(
                INFERENCE_MODULE_TEMPLATE.format(
                    input_signature=_get_input_signature(config),
                    input_dict=_get_input_dict(config),
                    output_tuple=_get_output_tuple(config),
                )
            )

        spec = importlib.util.spec_from_file_location("generated.ts", ts_path)
        gen_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_ts)

        gen_module = gen_ts.GeneratedInferenceModule(inference_module)
        scripted_module = torch.jit.script(gen_module)
    return scripted_module


def _get_type_map(dtype: str) -> str:
    """Return the Triton API type mapped to numpy type."""
    # see: https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
    return {
        "bool": "BOOL",
        "uint8": "UINT8",
        "uint16": "UINT16",
        "uint32": "UINT32",
        "uint64": "UINT64",
        "int8": "INT8",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
        "float16": "FP16",
        "float32": "FP32",
        "float64": "FP64",
        "string": "BYTES",
    }[dtype]


def _get_input_spec(model: LudwigModel) -> str:
    spec = []
    for feature_name, feature in model.model.input_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            TRITON_SPEC.format(
                prefix="INPUT",
                index=len(spec),
                name=feature.feature_name,
                data_type=_get_type_map(feature.get_preproc_input_dtype(metadata)),
                data_dims=1,  # hard code for now
            )
        )
    return ",".join(spec)


def _get_output_spec(model: LudwigModel) -> str:
    spec = []
    for feature_name, feature in model.model.output_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            TRITON_SPEC.format(
                prefix="OUTPUT",
                index=len(spec),
                name=feature.feature_name,  # Just output the one predictions column
                data_type=_get_type_map(feature.get_postproc_output_dtype(metadata)),
                data_dims=1,
            )
        )
    return ",".join(spec)


def _get_instance_spec(count=1, kind="KIND_CPU") -> str:
    spec = INSTANCE_SPEC.format(count=count, kind=kind)
    return spec


def _get_model_config(model: LudwigModel) -> str:
    config = TRITON_CONFIG_TEMPLATE.format(
        input_spec=_get_input_spec(model), output_spec=_get_output_spec(model), instance_spec=_get_instance_spec()
    )
    return config


def export_triton(
    model: LudwigModel, output_path: str, model_name: str = "ludwig_model", model_version: Union[int, str] = 1
) -> (str, str):
    """Exports a torchscript model to a output path that serves as a repository for Triton Inference Server.

    # Inputs

    :param model: (LudwigModel) A ludwig model.
    :param output_path: (str) The output path for the model repository.
    :param model_name: (str) The optional model name.
    :param model_name: (Union[int,str]) The optional model verison.

    # Return
    :return: (str, str) The saved model path, and config path.
    """
    model_ts = generate_triton_torchscript(model)
    model_dir = os.path.join(output_path, model_name, str(model_version))
    os.makedirs(model_dir, exist_ok=True)
    # Save the file to <model_repository>/<model_name>/<model_version>/model.pt
    model_path = os.path.join(model_dir, "model.pt")
    model_ts.save(model_path)
    # Save the default onfig to <model_repository>/<model_name>/config.pbtxt
    config_path = os.path.join(output_path, model_name, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(_get_model_config(model))
    return model_path, config_path
