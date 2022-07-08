import importlib.util
import os
import tempfile
from typing import Any, Dict, Tuple, Union, List

import torch
import pandas as pd

from dataclasses import dataclass
from ludwig.api import LudwigModel
from ludwig.constants import NAME
from ludwig.models.inference import InferenceModule, PREPROCESSOR, PREDICTOR, POSTPROCESSOR, INFERENCE_STAGES
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

TRITON_PREPROCESS_CONFIG_TEMPLATE = """
name: "ensemble_preprocess"
platform: "pytorch_libtorch"
max_batch_size: 0
input [{input_spec}
]
output [{output_spec}
]
instance_group [{instance_spec}
]
"""



class EnsemblePreprocessingConfig:
    pass



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


def single_to_input(s: pd.Series, feature) -> Union[List[str], torch.Tensor]:
    """Transform input for a feature to be compatible with what's required by TorchScript.
    """
    if s.dtype == "object" or type(feature) is CategoryInputFeature:
        return s.astype('str').to_list()
    return torch.from_numpy(s.to_numpy())

def all_to_input(df: pd.DataFrame, model: LudwigModel):
    """Transform input for all features to be compatible with what's required by TorchScript.
    """
    return {name: single_to_input(df[feature.column], feature) for name, feature in model.model.input_features.items()}


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
    order: int

    def __post_init__(self):
        # get Triton type
        self.type = to_triton_type(self.content)
        # get dimension
        self.dimension = to_triton_dimension(self.content)
        # get ensemble_scheduling output_map key (same as "name" in input/output)
        self.key = f"{self.kind}__{self.order}"
        # get ensemble_scheduling output_map value
        self.value = f"{self.name}_{self.inference_stage}"
        pass

@dataclass
class TritonEnsembleConfig:
    """
    will store triton config template, call the proper functions to populate it.
    could store path and have a function for save.
    """
    config_template: str
    pass

@dataclass
class TritonConfig:
    """
    will store triton config template, call the proper functions to populate it.
    could store path and have a function for save.
    """

    config_template: str
    pass

@dataclass
class TritonModel:
    """
    will store wrapper class template, call the proper functions to populate it.
    will return torchscript of each part of the inference pipeline.
    could store path and have a function for save.
    """
    wrapper_template: str
    pass

@dataclass
class TritonMaster: # change name
    input_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]]
    output_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]]
    inference_stage: str

    def __post_init__(self):
        if self.inference_stage in INFERENCE_STAGES:
            self.input_features = [TritonConfigFeature(feature_name, content, self.inference_stage, INPUT, i) for i, (feature_name, content) in
                                   enumerate(self.input_data_example.items())]
            self.output_features = [TritonConfigFeature(feature_name, content, self.inference_stage, OUTPUT, i) for i, (feature_name, content) in
                                   enumerate(self.output_data_example.items())]
        else:
            raise ValueError("Invalid inference stage. Choose one of {}".format(INFERENCE_STAGES))

def export_triton(
    model: LudwigModel, data_example: pd.DataFrame, output_path: str, model_name: str = "ludwig_model", model_version: Union[int, str] = 1
) -> Tuple[str, str]:
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
    data_example = all_to_input(data_example.head(1), model)

    preprocessor_output = inference_module.preprocessor_forward(data_example)
    preprocessor_triton_config = TritonMaster(data_example, preprocessor_output, PREPROCESSOR)

    predictor_output = inference_module.predictor_forward(preprocessor_output)
    predictor_triton_config = TritonMaster(preprocessor_output, predictor_output, PREDICTOR)

    postprocessor_output = inference_module.postprocessor_forward(predictor_output)
    postprocessor_triton_config = TritonMaster(predictor_output, postprocessor_output, POSTPROCESSOR)




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
