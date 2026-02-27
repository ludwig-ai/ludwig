import asyncio
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any

import torch

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import NAME
from ludwig.types import ModelConfigDict
from ludwig.utils.fs_utils import open_file

logger = logging.getLogger(__name__)


INFERENCE_MODULE_TEMPLATE = """
from typing import Any, Dict, List, Tuple, Union
import torch
from ludwig.utils.types import TorchscriptPreprocessingInput

class GeneratedInferenceModule(torch.nn.Module):
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module

    def forward(self, inputs: Dict[str, Any]):
        retyped_inputs: Dict[str, TorchscriptPreprocessingInput] = {{}}
        for k, v in inputs.items():
            assert isinstance(v, TorchscriptPreprocessingInput)
            retyped_inputs[k] = v

        results = self.inference_module(retyped_inputs)
        return {output_dicts}
"""


def _get_output_dicts(config: ModelConfigDict) -> str:
    results = []
    for feature in config["output_features"]:
        name = feature[NAME]
        results.append(f'"{name}": results["{name}"]["predictions"]')
    return "{" + ", ".join(results) + "}"


@DeveloperAPI
def generate_carton_torchscript(model: LudwigModel):
    config = model.config
    inference_module = model.to_torchscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "generated.py")
        with open_file(ts_path, "w") as f:
            f.write(
                INFERENCE_MODULE_TEMPLATE.format(
                    output_dicts=_get_output_dicts(config),
                )
            )

        spec = importlib.util.spec_from_file_location("generated.ts", ts_path)
        gen_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_ts)

        gen_module = gen_ts.GeneratedInferenceModule(inference_module)
        scripted_module = torch.jit.script(gen_module)
    return scripted_module


def _get_input_spec(model: LudwigModel) -> list[dict[str, Any]]:
    from cartonml import TensorSpec

    spec = []
    for feature_name, feature in model.model.input_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            TensorSpec(
                name=feature.feature_name, dtype=feature.get_preproc_input_dtype(metadata), shape=("batch_size",)
            )
        )
    return spec


def _get_output_spec(model: LudwigModel) -> list[dict[str, Any]]:
    from cartonml import TensorSpec

    spec = []
    for feature_name, feature in model.model.output_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append(
            TensorSpec(
                name=feature.feature_name, dtype=feature.get_postproc_output_dtype(metadata), shape=("batch_size",)
            )
        )
    return spec


@DeveloperAPI
def export_carton(model: LudwigModel, carton_path: str, carton_model_name="ludwig_model"):
    try:
        import cartonml as carton
    except ImportError:
        raise RuntimeError('The "cartonml-nightly" package is not installed in your environment.')

    # Generate a torchscript model
    model_ts = generate_carton_torchscript(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the model to a temp dir
        input_model_path: str = os.path.join(tmpdir, "model.pt")
        torch.jit.save(model_ts, input_model_path)

        # carton.pack is an async function so we run it and wait until it's complete
        # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
        # in another function
        async def pack() -> str:
            try:
                return await carton.pack(
                    path=input_model_path,
                    runner_name="torchscript",
                    # Any 2.x.x version is okay
                    # TODO: improve this
                    required_framework_version="=2",
                    model_name=carton_model_name,
                    inputs=_get_input_spec(model),
                    outputs=_get_output_spec(model),
                )
            except Exception as e:
                exception_message: str = 'An Exception inside "pack()" occurred.\n'
                exception_traceback: str = traceback.format_exc()
                exception_message += f'{type(e).__name__}: "{str(e)}".  Traceback: "{exception_traceback}".'
                sys.stderr.write(exception_message)
                sys.stderr.flush()
                raise ValueError(exception_message) from e  # Re-raise error for calling function to handle.

        try:
            tmp_out_path: str = asyncio.run(pack())
            # Move it to the output path
            shutil.move(tmp_out_path, carton_path)
        except Exception as e:
            exception_message: str = 'An Exception inside "export_carton()" occurred.\n'
            exception_traceback: str = traceback.format_exc()
            exception_message += f'{type(e).__name__}: "{str(e)}".  Traceback: "{exception_traceback}".'
            sys.stderr.write(exception_message)
            sys.stderr.flush()
            raise SystemExit(exception_message) from e  # Make sure error is fatal.
