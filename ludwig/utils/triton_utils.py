import importlib.util
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.features.category_feature import CategoryInputFeature
from ludwig.models.inference import (
    _InferencePostprocessor,
    _InferencePredictor,
    _InferencePreprocessor,
    InferenceModule,
    POSTPROCESSOR,
    PREDICTOR,
    PREPROCESSOR,
)
from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe
from ludwig.utils.torch_utils import DEVICE
from ludwig.utils.types import TorchAudioTuple, TorchscriptPreprocessingInput

INFERENCE_STAGES = [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]
INPUT = "INPUT"
OUTPUT = "OUTPUT"
ENSEMBLE = "ensemble"

INFERENCE_MODULE_TEMPLATE = """
from typing import Any, Dict, List, Union
import torch
from ludwig.utils.types import TorchscriptPreprocessingInput

class GeneratedInferenceModule(torch.nn.Module):
    def __init__(self, inference_module):
        super().__init__()
        self.inference_module = inference_module

    def forward(self, {input_signature}):
        with torch.no_grad():
            inputs: Dict[str, {input_type}] = {input_dict}
            results = self.inference_module(inputs)
            return {output_tuple}
"""

TRITON_SPEC = """
    {{
        name: "{key}" # "{name}"
        data_type: {data_type}
        dims: [ {data_dims} ]
    }}"""

INSTANCE_SPEC = """
    {{
        count: {count}
        kind: {kind}
    }}"""

TRITON_CONFIG_TEMPLATE = """name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 0 # Disable dynamic batching?
input [{input_spec}
]
output [{output_spec}
]
instance_group [{instance_spec}
]
"""

ENSEMBLE_SCHEDULING_INPUT_MAP = """
      input_map {{
        key: "{key}"
        value: "{value}"
      }}"""
ENSEMBLE_SCHEDULING_OUTPUT_MAP = """
      output_map {{
        key: "{key}"
        value: "{value}"
      }}"""

ENSEMBLE_SCHEDULING_STEP = """
    {{
      model_name: "{ensemble_model_name}"
      model_version: -1
      {input_maps}
      {output_maps}
    }}"""

TRITON_ENSEMBLE_CONFIG_TEMPLATE = """name: "{model_name}"
platform: "ensemble"
max_batch_size: 0 # Dynamic batch not supported
input [{input_spec}
]
output [{output_spec}
]
ensemble_scheduling {{
  step [{ensemble_scheduling_steps}
  ]
}}
"""


def _get_type_map(dtype: str) -> str:
    """Return the Triton API type mapped to numpy type."""
    # see: https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
    return {
        "bool": "TYPE_BOOL",
        "uint8": "TYPE_UINT8",
        "uint16": "TYPE_UINT16",
        "uint32": "TYPE_UINT32",
        "uint64": "TYPE_UINT64",
        "int8": "TYPE_INT8",
        "int16": "TYPE_INT16",
        "int32": "TYPE_INT32",
        "int64": "TYPE_INT64",
        "float16": "TYPE_FP16",
        "float32": "TYPE_FP32",
        "float64": "TYPE_FP64",
        "string": "TYPE_STRING",
        "torch.float32": "TYPE_FP32",
        "torch.float": "TYPE_FP32",
        "torch.float64": "TYPE_FP64",
        "torch.double": "TYPE_FP64",
        "torch.float16": "TYPE_FP16",
        "torch.half": "TYPE_FP16",
        "torch.uint8": "TYPE_UINT8",
        "torch.int8": "TYPE_INT8",
        "torch.int16": "TYPE_INT16",
        "torch.short": "TYPE_INT16",
        "torch.int32": "TYPE_INT32",
        "torch.int": "TYPE_INT32",
        "torch.int64": "TYPE_INT64",
        "torch.long": "TYPE_INT64",
        "torch.bool": "TYPE_BOOL",
    }[dtype]


def to_triton_dimension(content: Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]):
    if isinstance(content, list) and content:
        if isinstance(content[0], str):
            return [len(content)]
        """
        todo (Wael): check these and add other types

        if isinstance(content[0], torch.Tensor):
            return (len(content))
        if isinstance(content[0], TorchAudioTuple):
            return (len(content))
        """
    elif isinstance(content, torch.Tensor):
        return list(content.size())
    return [-1]


def to_triton_type(content: Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]):
    if isinstance(content, list) and content:
        if isinstance(content[0], str):
            return _get_type_map("string")
        """
        todo (Wael): check these and add other types

        if isinstance(content[0], torch.Tensor):
            return _get_type_map(content.dtype)
        if isinstance(content[0], TorchAudioTuple):
            return _get_type_map(content.dtype)
        """
    elif isinstance(content, torch.Tensor):
        return _get_type_map(str(content.dtype))


@dataclass
class TritonConfigFeature:
    """Represents an input/output feature in a Triton config.

    :param name: name of the feature.
    :param content: the data contents of the feature.
    :param inference_stage: one of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    :param kind: one of INPUT, OUTPUT.
    :param index: index of the feature in the Triton Config.
    """

    name: str
    content: Union[TorchscriptPreprocessingInput, torch.Tensor]
    inference_stage: str
    kind: str
    index: int

    def __post_init__(self):
        # removing non-alphanumeric characters as this will go in the wrapper function header.
        self.wrapper_signature_name = re.sub(r"[\W]+", "_", self.name)
        # get Triton type
        self.type = to_triton_type(self.content)
        # get dimension
        self.dimension = to_triton_dimension(self.content)
        # get ensemble_scheduling output_map key (same as "name" in input/output)
        self.key = f"{self.kind}__{self.index}"

        # get ensemble_scheduling output_map value.
        if self.inference_stage == PREDICTOR and self.kind == INPUT:
            # PREPROCESSOR outputs and PREDICTOR inputs must have the same "value" attribute.
            self.value = f"{PREPROCESSOR}_{OUTPUT}_{self.index}"
        elif self.inference_stage == POSTPROCESSOR and self.kind == INPUT:
            # PREDICTOR outputs and POSTPROCESSOR inputs must have the same "value" attribute.
            self.value = f"{PREDICTOR}_{OUTPUT}_{self.index}"
        else:
            self.value = f"{self.inference_stage}_{self.kind}_{self.index}"


@dataclass
class TritonMaster:
    """Provides access to the Triton Config and the scripted module.

    :param module: the inference module.
    :param input_data_example: an input for the module that will help determine the
        input and output dimensions.
    :param inference_stage: one of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    """

    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]
    input_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]]
    inference_stage: str
    model_name: str
    output_path: str
    model_version: int

    def __post_init__(self):
        """Extract input and output features and necessary information for a Triton config."""
        if self.inference_stage not in INFERENCE_STAGES:
            raise ValueError(f"Invalid inference stage. Choose one of {INFERENCE_STAGES}")

        self.full_model_name = self.model_name + "_" + self.inference_stage
        self.base_path = os.path.join(self.output_path, self.full_model_name)
        os.makedirs(self.base_path, exist_ok=True)

        # get output for sample input.
        self.output_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]] = self.module(
            self.input_data_example
        )

        # generate input and output features.
        self.input_features: List[TritonConfigFeature] = [
            TritonConfigFeature(feature_name, content, self.inference_stage, INPUT, i)
            for i, (feature_name, content) in enumerate(self.input_data_example.items())
        ]
        self.output_features: List[TritonConfigFeature] = [
            TritonConfigFeature(feature_name, content, self.inference_stage, OUTPUT, i)
            for i, (feature_name, content) in enumerate(self.output_data_example.items())
        ]

    def save_model(self) -> str:
        """Scripts the model and saves it."""
        if not isinstance(self.model_version, int) or self.model_version < 1:
            raise ValueError("Model version has to be a non-zero positive integer")
        pass

        os.makedirs(os.path.join(self.base_path, str(self.model_version)), exist_ok=True)
        model_path = os.path.join(self.base_path, str(self.model_version), "model.pt")
        self.model_ts = TritonModel(
            self.module, self.input_features, self.output_features, self.inference_stage
        ).generate_scripted_module()
        self.model_ts.save(model_path)
        return model_path

    def save_config(self) -> str:
        """Save the Triton config."""
        self.config = TritonConfig(self.full_model_name, self.input_features, self.output_features)
        config_path = os.path.join(self.base_path, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(self.config.get_model_config())
        return config_path


@dataclass
class TritonEnsembleConfig:
    """Dataclass for creating and saving the Triton ensemble config."""

    triton_master_preprocessor: TritonMaster
    triton_master_predictor: TritonMaster
    triton_master_postprocessor: TritonMaster
    model_name: str
    output_path: str

    def __post_init__(self):
        self.ensemble_model_name = self.model_name + "_" + ENSEMBLE
        self.base_path = os.path.join(self.output_path, self.ensemble_model_name)
        os.makedirs(self.base_path, exist_ok=True)

    def _get_ensemble_scheduling_input_maps(self, triton_features: List[TritonConfigFeature]) -> str:
        return "".join(
            ENSEMBLE_SCHEDULING_INPUT_MAP.format(key=feature.key, value=feature.value) for feature in triton_features
        )

    def _get_ensemble_scheduling_output_maps(self, triton_features: List[TritonConfigFeature]) -> str:
        return "".join(
            ENSEMBLE_SCHEDULING_OUTPUT_MAP.format(key=feature.key, value=feature.value) for feature in triton_features
        )

    def _get_ensemble_scheduling_step(self, triton_master: TritonMaster):
        return ENSEMBLE_SCHEDULING_STEP.format(
            ensemble_model_name=triton_master.config.full_model_name,
            input_maps=self._get_ensemble_scheduling_input_maps(triton_master.input_features),
            output_maps=self._get_ensemble_scheduling_output_maps(triton_master.output_features),
        )

    def _get_ensemble_spec(self, triton_features: List[TritonConfigFeature]) -> str:
        spec = []
        for feature in triton_features:
            spec.append(
                TRITON_SPEC.format(
                    key=feature.value,
                    name=feature.name,
                    data_type=feature.type,
                    data_dims=", ".join(str(dim) for dim in feature.dimension),  # check correctness
                )
            )
        return ",".join(spec)

    def get_config(self):
        triton_masters = [
            self.triton_master_preprocessor,
            self.triton_master_predictor,
            self.triton_master_postprocessor,
        ]
        ensemble_scheduling_steps = ",".join(
            [self._get_ensemble_scheduling_step(triton_master) for triton_master in triton_masters]
        )
        return TRITON_ENSEMBLE_CONFIG_TEMPLATE.format(
            model_name=self.ensemble_model_name,
            input_spec=self._get_ensemble_spec(self.triton_master_preprocessor.input_features),
            output_spec=self._get_ensemble_spec(self.triton_master_postprocessor.output_features),
            ensemble_scheduling_steps=ensemble_scheduling_steps,
        )

    def save_ensemble_config(self):
        config_path = os.path.join(self.base_path, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(self.get_config())
        return config_path


@dataclass
class TritonConfig:
    """Enables the creation and export of a Triton config.

    :param full_model_name: name of the model. Must be the same as the directory where the config is saved.
    :param input_features: input features of the model.
    :param output_features: output features of the model.
    """

    full_model_name: str
    input_features: List[TritonConfigFeature]
    output_features: List[TritonConfigFeature]

    def _get_triton_spec(self, triton_features: List[TritonConfigFeature]) -> str:
        spec = []
        for feature in triton_features:
            spec.append(
                TRITON_SPEC.format(
                    key=feature.key,
                    name=feature.name,
                    data_type=feature.type,
                    data_dims=", ".join(str(dim) for dim in feature.dimension),  # check correctness
                )
            )
        return ",".join(spec)

    def _get_instance_spec(self, count=1, kind="KIND_CPU") -> str:
        spec = INSTANCE_SPEC.format(count=count, kind=kind)
        return spec

    def get_model_config(self) -> str:
        """Generate a Triton config for a model from the input and output features."""
        config = TRITON_CONFIG_TEMPLATE.format(
            model_name=self.full_model_name,
            input_spec=self._get_triton_spec(self.input_features),
            output_spec=self._get_triton_spec(self.output_features),
            instance_spec=self._get_instance_spec(),
        )
        return config


@dataclass
class TritonModel:
    """Enables the scripting and export of a model.

    :param module: the inference module.
    :param input_features: input features of the model.
    :param output_features: output features of the model.
    :param inference_stage: one of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    """

    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]
    input_features: List[TritonConfigFeature]
    output_features: List[TritonConfigFeature]
    inference_stage: str

    def _get_wrapper_signature_type(self) -> str:
        return {
            PREPROCESSOR: "TorchscriptPreprocessingInput",
            PREDICTOR: "torch.Tensor",
            POSTPROCESSOR: "torch.Tensor",
        }[self.inference_stage]

    def _get_input_signature(self, triton_features: List[TritonConfigFeature]) -> str:
        signature_type = self._get_wrapper_signature_type()
        elems = [f"{feature.wrapper_signature_name}: {signature_type}" for feature in triton_features]
        return ", ".join(elems)

    def _get_input_dict(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [f'"{feature.name}": {feature.wrapper_signature_name}' for feature in triton_features]
        return "{" + ", ".join(elems) + "}"

    def _get_output_tuple(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [f'results["{feature.name}"]' for feature in triton_features]
        return "(" + ", ".join(elems) + ",)"

    def generate_inference_module_wrapper(self) -> str:
        """Generate the class wrapper around an inference module."""
        return INFERENCE_MODULE_TEMPLATE.format(
            input_signature=self._get_input_signature(self.input_features),
            input_type=self._get_wrapper_signature_type(),
            input_dict=self._get_input_dict(self.input_features),
            output_tuple=self._get_output_tuple(self.output_features),
        )

    def generate_scripted_module(self):
        """Generate the scripted module from the wrapper class."""
        wrapper_definition = self.generate_inference_module_wrapper()
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_path = os.path.join(tmpdir, "generated.py")
            with open(ts_path, "w") as f:
                f.write(wrapper_definition)

            spec = importlib.util.spec_from_file_location("generated.ts", ts_path)
            gen_ts = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_ts)

            gen_module = gen_ts.GeneratedInferenceModule(self.module)
            scripted_module = torch.jit.script(gen_module)
        return scripted_module


def export_triton(
    model: LudwigModel,
    data_example: pd.DataFrame,
    output_path: str = "model_repository",
    model_name: str = "ludwig_model",
    model_version: Union[int, str] = 1,
) -> Dict[str, Tuple[str, str]]:
    """Exports a torchscript model to a output path that serves as a repository for Triton Inference Server.

    # Inputs
    :param model: (LudwigModel) A ludwig model.
    :param data_example: (pd.DataFrame) an example from the dataset.
        Used to get dimensions throughout the pipeline.
    :param output_path: (str) The output path for the model repository.
    :param model_name: (str) The optional model name.
    :param model_version: (Union[int,str]) The optional model verison.

    # Return
    :return: (str, str) The saved model path, and config path.
    """

    inference_module = InferenceModule.from_ludwig_model(model.model, model.config, model.training_set_metadata, DEVICE)
    split_modules = [inference_module.preprocessor, inference_module.predictor, inference_module.postprocessor]
    example_input = to_inference_module_input_from_dataframe(data_example.head(10), model.config, load_paths=True)
    paths = {}
    triton_masters = []
    for i, module in enumerate(split_modules):
        triton_master = TritonMaster(module, example_input, INFERENCE_STAGES[i], model_name, output_path, model_version)
        example_input = triton_master.output_data_example

        config_path = triton_master.save_config()
        model_path = triton_master.save_model()
        paths[INFERENCE_STAGES[i]] = (config_path, model_path)
        triton_masters.append(triton_master)

    # saving ensemble config
    triton_master_preprocessor, triton_master_predictor, triton_master_postprocessor = triton_masters
    ensemble_config = TritonEnsembleConfig(
        triton_master_preprocessor, triton_master_predictor, triton_master_postprocessor, model_name, output_path
    )
    ensemble_config_path = ensemble_config.save_ensemble_config()
    paths[ENSEMBLE] = (ensemble_config_path,)

    return paths
