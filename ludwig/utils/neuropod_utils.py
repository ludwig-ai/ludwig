import importlib.util
import logging
import os
import tempfile
from typing import Any, Dict, List

import torch

from ludwig.api import LudwigModel
from ludwig.constants import NAME

logger = logging.getLogger(__name__)


INFERENCE_MODULE_TEMPLATE = """
from typing import Any, Dict, List, Union
import torch

class GeneratedInferenceModule(torch.nn.Module):
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module

    def forward(self, {input_signature}):
        inputs = {input_dict}
        results = self.inference_module(inputs)
        return {output_dicts}
"""


def _get_input_signature(config: Dict[str, Any]) -> str:
    args = []
    for feature in config["input_features"]:
        name = feature[NAME]
        args.append(f"{name}: Union[List[str], List[torch.Tensor], torch.Tensor]")
    return ", ".join(args)


def _get_input_dict(config: Dict[str, Any]) -> str:
    elems = []
    for feature in config["input_features"]:
        name = feature[NAME]
        elems.append(f'"{name}": {name}')
    return "{" + ", ".join(elems) + "}"


def _get_output_dicts(config: Dict[str, Any]) -> str:
    results = []
    for feature in config["output_features"]:
        name = feature[NAME]
        results.append("{" + f'"{name}": results["{name}"]["predictions"]' + "}")
    return ", ".join(results)


def generate_neuropod_torchscript(model: LudwigModel):
    config = model.config
    inference_module = model.to_torchscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "generated.py")
        with open(ts_path, "w") as f:
            f.write(
                INFERENCE_MODULE_TEMPLATE.format(
                    input_signature=_get_input_signature(config),
                    input_dict=_get_input_dict(config),
                    output_dicts=_get_output_dicts(config),
                )
            )

        spec = importlib.util.spec_from_file_location("generated.ts", ts_path)
        gen_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_ts)

        gen_module = gen_ts.GeneratedInferenceModule(inference_module)
        scripted_module = torch.jit.script(gen_module)
    return scripted_module


def _get_input_spec(model: LudwigModel) -> List[Dict[str, Any]]:
    spec = []
    for feature_name, feature in model.model.input_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            {"name": feature.feature_name, "dtype": feature.get_preproc_input_dtype(metadata), "shape": ("batch_size",)}
        )
    return spec


def _get_output_spec(model: LudwigModel) -> List[Dict[str, Any]]:
    spec = []
    for feature_name, feature in model.model.output_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            {
                "name": feature.feature_name,
                "dtype": feature.get_postproc_output_dtype(metadata),
                "shape": ("batch_size",),
            }
        )
    return spec


def export_neuropod(model: LudwigModel, neuropod_path: str, neuropod_model_name="ludwig_model"):
    try:
        from neuropod.backends.torchscript.packager import create_torchscript_neuropod
    except ImportError:
        raise RuntimeError('The "neuropod" package is not installed in your environment.')

    model_ts = generate_neuropod_torchscript(model)
    create_torchscript_neuropod(
        neuropod_path=neuropod_path,
        model_name=neuropod_model_name,
        module=model_ts,
        input_spec=_get_input_spec(model),
        output_spec=_get_output_spec(model),
    )
