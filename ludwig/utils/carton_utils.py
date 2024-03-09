import asyncio
import importlib.util
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List

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


def _get_input_spec(model: LudwigModel) -> List[Dict[str, Any]]:
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


def _get_output_spec(model: LudwigModel) -> List[Dict[str, Any]]:
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
    print(f"\n[ALEX_TEST] [WOUTPUT] MODEL_TORCH_SCRIPT:\n{model_ts} ; TYPE: {str(type(model_ts))}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the model to a temp dir
        input_model_path: str = os.path.join(tmpdir, "model.pt")
        torch.jit.save(model_ts, input_model_path)

        # carton.pack is an async function so we run it and wait until it's complete
        # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
        # in another function
        # TODO: <Alex>ALEX</Alex>
        # async def pack():
        #     return await carton.pack(
        #         input_model_path,
        #         runner_name="torchscript",
        #         # Any 2.x.x version is okay
        #         # TODO: improve this
        #         required_framework_version="=2",
        #         model_name=carton_model_name,
        #         inputs=_get_input_spec(model),
        #         outputs=_get_output_spec(model),
        #     )

        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        async def packster() -> str:
            time.sleep(1)
            # TODO: <Alex>ALEX</Alex>
            # try:
            #     a: str = await carton.pack(
            #         input_model_path,
            #         runner_name="torchscript",
            #         # Any 2.x.x version is okay
            #         # TODO: improve this
            #         required_framework_version="=2",
            #         model_name=carton_model_name,
            #         inputs=_get_input_spec(model),
            #         outputs=_get_output_spec(model),
            #     )
            #     time.sleep(1)
            #     print(f"\n[ALEX_TEST] [WOUTPUT] WOUTPUT:\n{a} ; TYPE: {str(type(a))}")
            #     time.sleep(1)
            #     return a
            # except Exception as ie:
            #     exception_message: str = "A Packster-Inside Exception occurred.\n"
            #     exception_traceback: str = traceback.format_exc()
            #     exception_message += f'{type(ie).__name__}: "{str(ie)}".  Traceback: "{exception_traceback}".'
            #     sys.stderr.write(exception_message)
            #     sys.stderr.flush()
            #     raise ValueError(exception_message) from ie
            # TODO: <Alex>ALEX</Alex>
            # TODO: <Alex>ALEX</Alex>
            max_tries: int = 5
            idx: int
            for idx in range(max_tries):
                print(f"\n[ALEX_TEST] [WOUTPUT] TRYING_IDX:\n{idx} ; TYPE: {str(type(idx))}")
                try:
                    a: str = await carton.pack(
                        input_model_path,
                        runner_name="torchscript",
                        # Any 2.x.x version is okay
                        # TODO: improve this
                        required_framework_version="=2",
                        model_name=carton_model_name,
                        inputs=_get_input_spec(model),
                        outputs=_get_output_spec(model),
                    )
                    time.sleep(1)
                    print(f"\n[ALEX_TEST] [WOUTPUT] WOUTPUT:\n{a} ; TYPE: {str(type(a))}")
                    time.sleep(1)
                    return a
                except Exception as ie:
                    exception_message: str = "A Packster-Inside Exception occurred.\n"
                    exception_traceback: str = traceback.format_exc()
                    exception_message += f'{type(ie).__name__}: "{str(ie)}".  Traceback: "{exception_traceback}".'
                    sys.stderr.write(exception_message)
                    sys.stderr.flush()
                    # raise ValueError(exception_message) from ie
                if idx >= max_tries - 1:
                    raise ValueError(exception_message) from ie
            # TODO: <Alex>ALEX</Alex>

        # TODO: <Alex>ALEX</Alex>

        # TODO: <Alex>ALEX</Alex>
        loop = asyncio.get_event_loop()
        print(f"\n[ALEX_TEST] [WOUTPUT] LOOP:\n{loop} ; TYPE: {str(type(loop))}")
        # tmp_out_path = loop.run_until_complete(pack())
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        import time
        import sys
        import traceback

        try:
            # TODO: <Alex>ALEX</Alex>
            # tmp_out_path = loop.run_until_complete(pack())
            # TODO: <Alex>ALEX</Alex>
            # TODO: <Alex>ALEX</Alex>
            time.sleep(1)
            tmp_out_path: str = loop.run_until_complete(packster())
            # TODO: <Alex>ALEX</Alex>
        except Exception as e:
            exception_message: str = "A general Exception occurred.\n"
            exception_traceback: str = traceback.format_exc()
            exception_message += f'{type(e).__name__}: "{str(e)}".  Traceback: "{exception_traceback}".'
            sys.stderr.write(exception_message)
            sys.stderr.flush()
            raise SystemExit(exception_message) from e  # Make sure error is fatal.
        # TODO: <Alex>ALEX</Alex>

        # Move it to the output path
        time.sleep(1)
        shutil.move(tmp_out_path, carton_path)
