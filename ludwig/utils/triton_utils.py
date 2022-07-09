import importlib.util
import os
import re
import tempfile
from typing import Dict, Tuple, Union, List

import torch
import pandas as pd

from dataclasses import dataclass
from ludwig.api import LudwigModel
from ludwig.models.inference import InferenceModule, _InferencePreprocessor, _InferencePredictor, \
    _InferencePostprocessor, INFERENCE_STAGES
from ludwig.features.category_feature import CategoryInputFeature
from ludwig.utils.torch_utils import DEVICE
from ludwig.utils.types import TorchscriptPreprocessingInput, TorchAudioTuple

INPUT = "INPUT"
OUTPUT = "OUTPUT"

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
        name: "{key}" # "{name}"
        data_type: {data_type}
        dims: [ {data_dims} ]
    }}"""

INSTANCE_SPEC = """
    {{
        count: {count}
        kind: {kind}
    }}"""

TRITON_CONFIG_TEMPLATE = """
name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 0 # Disable dynamic batching?
input [{input_spec}
]
output [{output_spec}
]
instance_group [{instance_spec}
]
"""


def _get_type_map(dtype: str) -> str:
    """Return the Triton API type mapped to numpy type.

    todo (Wael): add pytorch types and what they correspond to.
    """
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


def raw_feature_to_inference_input(s: pd.Series, feature) -> Union[List[str], torch.Tensor]:
    """Transform input for a feature to be compatible with what's required by TorchScript.
    """
    if s.dtype == "object" or type(feature) is CategoryInputFeature:
        return s.astype('str').to_list()
    return torch.from_numpy(s.to_numpy())


def raw_to_inference_input(df: pd.DataFrame, model: LudwigModel):
    """Transform input for all features to be compatible with what's required by TorchScript.
    """
    return {name: raw_feature_to_inference_input(df[feature.column], feature) for name, feature in
            model.model.input_features.items()}


def to_triton_dimension(content: Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]):
    if isinstance(content, list) and content:
        if isinstance(content[0], str):
            return (len(content))
        """
        todo (Wael): check these and add other types
        
        if isinstance(content[0], torch.Tensor):
            return (len(content))
        if isinstance(content[0], TorchAudioTuple):
            return (len(content))
        """
    elif isinstance(content, torch.Tensor):
        return tuple(content.size())


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
        return _get_type_map(content.dtype)


@dataclass
class TritonConfigFeature:
    name: str
    content: Union[TorchscriptPreprocessingInput, torch.Tensor]
    inference_stage: str
    kind: str
    index: int

    def __post_init__(self):
        # removing non-alphanumeric characters as this will go in the wrapper function header.
        self.wrapper_signature_name = re.sub(r'[\W]+', '_', self.name)
        # get Triton type
        self.type = to_triton_type(self.content)
        # get dimension
        self.dimension = to_triton_dimension(self.content)
        # get ensemble_scheduling output_map key (same as "name" in input/output)
        self.key = f"{self.kind}__{self.index}"
        # get ensemble_scheduling output_map value
        self.value = f"{self.name}_{self.inference_stage}_{self.kind}"


@dataclass
class TritonEnsembleConfig:
    """
    will store triton config template, call the proper functions to populate it.
    could store path and have a function for save.
    """
    config_template: str
    pass


@dataclass
class TritonMaster:  # change name
    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]
    input_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]]
    inference_stage: str

    def __post_init__(self):
        """Extract input and output features and necessary information for a Triton config.
        """
        if self.inference_stage not in INFERENCE_STAGES:
            raise ValueError("Invalid inference stage. Choose one of {}".format(INFERENCE_STAGES))

        self.output_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]] = self.module(
            self.input_data_example)
        self.input_features: List[TritonConfigFeature] = [
            TritonConfigFeature(feature_name, content, self.inference_stage, INPUT, i) for i, (feature_name, content) in
            enumerate(self.input_data_example.items())]
        self.output_features: List[TritonConfigFeature] = [
            TritonConfigFeature(feature_name, content, self.inference_stage, OUTPUT, i) for i, (feature_name, content)
            in
            enumerate(self.output_data_example.items())]

    def get_inference_module(self):
        pass

    def save_model(self, path, version=1) -> str:
        if not isinstance(version, int) or version < 1:
            raise ValueError("Model version has to be a non-zero positive integer")
        pass
        model_path = os.path.join(path, self.inference_stage, str(version), "model.pt")
        model_ts = TritonModel(self.module, self.output_features, self.output_features).generate_scripted_module()
        model_ts.save(model_path)
        return model_path

    def save_config(self, path: str, full_model_name: str) -> str:
        """Save the Triton config to path
        """
        config = TritonConfig(full_model_name, self.input_features, self.output_features)
        config_path = os.path.join(path, self.inference_stage, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(config.get_model_config())
        return config_path


@dataclass
class TritonConfig:
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
        """Generate a Triton config for a model from the input and output features.

        todo (Wael): add parameters to _get_instance_spec
        """
        config = TRITON_CONFIG_TEMPLATE.format(
            model_name=self.full_model_name,
            input_spec=self._get_triton_spec(self.input_features),
            output_spec=self._get_triton_spec(self.output_features),
            instance_spec=self._get_instance_spec()
        )
        return config


@dataclass
class TritonModel:
    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]
    triton_input_features: List[TritonConfigFeature]
    triton_output_features: List[TritonConfigFeature]

    def _get_input_signature(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [f"{feature.wrapper_signature_name}: TorchscriptPreprocessingInput" for feature in triton_features]
        return ", ".join(elems)

    def _get_input_dict(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [f'"{feature.name}": {feature.wrapper_signature_name}' for feature in triton_features]
        return "{" + ", ".join(elems) + "}"

    def _get_output_tuple(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [f'results["{feature.name}"]' for feature in triton_features]
        return "(" + ", ".join(elems) + ")"

    def generate_inference_module_wrapper(self) -> str:
        return INFERENCE_MODULE_TEMPLATE.format(
            input_signature=self._get_input_signature(self.triton_input_features),
            input_dict=self._get_input_dict(self.triton_input_features),
            output_tuple=self._get_output_tuple(self.triton_output_features),
        )

    def generate_scripted_module(self):
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


def export_triton(model: LudwigModel, data_example: pd.DataFrame, output_path: str = "model_repository",
                  model_name: str = "ludwig_model", model_version: Union[int, str] = 1) -> Dict[str, Tuple[str, str]]:
    """Exports a torchscript model to a output path that serves as a repository for Triton Inference Server.

    # Inputs
    :param model: (LudwigModel) A ludwig model.
    :param data_example: (pd.DataFrame) an example from the dataset.
        Used to get dimensions throughout the pipeline.
    :param output_path: (str) The output path for the model repository.
    :param model_name: (str) The optional model name.
    :param model_name: (Union[int,str]) The optional model verison.

    # Return
    :return: (str, str) The saved model path, and config path.
    """

    inference_module = InferenceModule.from_ludwig_model(model.model, model.config, model.training_set_metadata, DEVICE)
    example_input = raw_to_inference_input(data_example.head(1), model)
    paths = {}
    for i, module in enumerate(
            [inference_module.preprocessor, inference_module.predictor, inference_module.postprocessor]):
        triton_master = TritonMaster(module, example_input, INFERENCE_STAGES[i])
        example_input = triton_master.output_data_example

        full_model_name = model_name + "_" + INFERENCE_STAGES[i]
        base_path = os.path.join(output_path, full_model_name)
        os.makedirs(base_path, exist_ok=True)

        config_path = triton_master.save_config(base_path, full_model_name)
        model_path = triton_master.save_model(path=base_path, version=model_version)
        paths[INFERENCE_STAGES[i]] = (config_path, model_path)

    return paths
