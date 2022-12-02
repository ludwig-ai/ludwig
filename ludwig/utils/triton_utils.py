import importlib.util
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    IMAGE,
    INPUT_FEATURES,
    POSTPROCESSOR,
    PREDICTOR,
    PREPROCESSOR,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    TYPE,
    VECTOR,
)
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.models.inference import (
    _InferencePostprocessor,
    _InferencePredictor,
    _InferencePreprocessor,
    InferenceModule,
)
from ludwig.types import ModelConfigDict
from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe
from ludwig.utils.misc_utils import remove_empty_lines
from ludwig.utils.torch_utils import model_size, place_on_device
from ludwig.utils.types import TorchAudioTuple, TorchscriptPreprocessingInput

FEATURES_TO_CAST_AS_STRINGS = {BINARY, CATEGORY, BAG, SET, TEXT, SEQUENCE, TIMESERIES, VECTOR}

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

FEATURE_RESHAPE_SPEC = """reshape: {{ shape: [ {reshape_dims} ] }}
"""

TRITON_SPEC = """
    {{
        name: "{key}"
        data_type: {data_type}
        dims: [ {data_dims} ]
        {reshape_spec}
    }}"""

INSTANCE_SPEC = """
    {{
        count: {count}
        kind: {kind}
    }}"""

DYNAMIC_BATCHING_TEMPLATE = """dynamic_batching {{
    max_queue_delay_microseconds: {delay}
}}"""

TRITON_CONFIG_TEMPLATE = """name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: {max_batch_size}
{dynamic_batching_spec}
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
max_batch_size: 0
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
    # todo (Wael): tests for all types.
    if isinstance(content, list) and content:
        if isinstance(content[0], str):
            return [len(content)]
    elif isinstance(content, torch.Tensor):
        return list(content.size())
    return [-1]


def to_triton_type(content: Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]):
    # todo (Wael): tests for all types.
    if isinstance(content, list) and content:
        if isinstance(content[0], str):
            return _get_type_map("string")
    elif isinstance(content, torch.Tensor):
        return _get_type_map(str(content.dtype))


@DeveloperAPI
@dataclass
class TritonArtifact:
    """Dataclass for exported Triton artifacts."""

    # Name of the model.
    model_name: str

    # Model version.
    model_version: Union[int, str]

    # Triton backend (e.g. "pytorch_libtorch").
    platform: str

    # Model path.
    path: str

    # Type of artifact (application/octet-stream, plain text, etc.)
    content_type: str

    # Size of the artifact in bytes.
    content_length: int


@DeveloperAPI
@dataclass
class TritonConfigFeature:
    """Represents an input/output feature in a Triton config."""

    # Name of the feature.
    name: str

    # Ludwig type of the feature, or "tensor"
    ludwig_type: str

    # The data contents of the feature.
    content: Union[TorchscriptPreprocessingInput, torch.Tensor]

    # One of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    inference_stage: str

    # One of INPUT, OUTPUT.
    kind: str

    # Index of the feature in the Triton Config.
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
        self.value = self._get_feature_ensemble_value()

    def _get_feature_ensemble_value(self) -> str:
        # get ensemble_scheduling output_map value.
        if self.inference_stage == PREPROCESSOR and self.kind == INPUT:
            return self.name
        if self.inference_stage == PREDICTOR and self.kind == INPUT:
            # PREPROCESSOR outputs and PREDICTOR inputs must have the same "value" attribute.
            return f"{PREPROCESSOR}_{OUTPUT}_{self.index}"
        elif self.inference_stage == POSTPROCESSOR and self.kind == INPUT:
            # PREDICTOR outputs and POSTPROCESSOR inputs must have the same "value" attribute.
            return f"{PREDICTOR}_{OUTPUT}_{self.index}"
        elif self.inference_stage == POSTPROCESSOR and self.kind == OUTPUT:
            return self.name
        else:
            return f"{self.inference_stage}_{self.kind}_{self.index}"

    def _get_wrapper_signature_type(self) -> str:
        if self.ludwig_type in FEATURES_TO_CAST_AS_STRINGS:
            return "List[str]"
        elif self.ludwig_type in [IMAGE, AUDIO, DATE, "tensor"]:
            return {
                IMAGE: "List[torch.Tensor]",
                AUDIO: "TorchAudioTuple",
                DATE: "List[torch.Tensor]",
                "tensor": "torch.Tensor",
            }[self.ludwig_type]
        return "torch.Tensor"


@DeveloperAPI
@dataclass
class TritonMaster:
    """Provides access to the Triton Config and the scripted module."""

    # The inference module.
    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]

    # An input for the module that will help determine the input and output dimensions.
    input_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]]

    # One of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    inference_stage: str

    # Max batch size of the model. (Triton config param).
    max_batch_size: int

    # Max time a request to the Triton model spends in the queue. (Triton config param).
    max_queue_delay_microseconds: int

    # Name of the model as specified by the caller of the triton_export function.
    model_name: str

    # Base output directory. Corresponds to the Triton model registry.
    output_path: str

    # Triton model version.
    model_version: Union[int, str]

    # Ludwig config.
    ludwig_config: ModelConfigDict

    # One of "cpu", "cuda".
    device: str

    # Number of model instances on device.
    model_instance_count: int

    def __post_init__(self):
        """Extract input and output features and necessary information for a Triton config."""
        if self.inference_stage not in INFERENCE_STAGES:
            raise ValueError(f"Invalid inference stage. Choose one of {INFERENCE_STAGES}")

        self.full_model_name = self.model_name + "_" + self.inference_stage
        self.base_path = os.path.join(self.output_path, self.full_model_name)
        os.makedirs(self.base_path, exist_ok=True)

        self.output_data_example: Dict[str, Union[TorchscriptPreprocessingInput, torch.Tensor]] = self.module(
            self.input_data_example
        )

        # generate input and output features.
        self.input_features: List[TritonConfigFeature] = []
        for i, (feature_name, content) in enumerate(self.input_data_example.items()):
            ludwig_type = "tensor"
            if self.inference_stage == PREPROCESSOR:
                ludwig_type = self.ludwig_config[INPUT_FEATURES][i][TYPE]
            self.input_features.append(
                TritonConfigFeature(feature_name, ludwig_type, content, self.inference_stage, INPUT, i)
            )

        self.output_features: List[TritonConfigFeature] = []
        for i, (feature_name, content) in enumerate(self.output_data_example.items()):
            ludwig_type = "tensor"
            self.output_features.append(
                TritonConfigFeature(feature_name, ludwig_type, content, self.inference_stage, OUTPUT, i)
            )

    def save_model(self) -> TritonArtifact:
        """Script the model and saves it.

        Return the appropriate artifact.
        """
        if isinstance(self.model_version, str) and self.model_version.isdigit():
            self.model_version = int(self.model_version)
        if not isinstance(self.model_version, int) or self.model_version < 1:
            raise ValueError("Model version has to be a non-zero positive integer")
        pass

        # wrapper.py is optional and is just for visualizing the inputs/outputs to the model exported to Triton.
        wrapper_definition = TritonModel(
            self.module, self.input_features, self.output_features, self.inference_stage
        ).generate_inference_module_wrapper()
        with open(os.path.join(self.base_path, "wrapper.py"), "w") as f:
            f.write(wrapper_definition)

        os.makedirs(os.path.join(self.base_path, str(self.model_version)), exist_ok=True)
        model_path = os.path.join(self.base_path, str(self.model_version), "model.pt")

        self.model_ts = TritonModel(
            self.module, self.input_features, self.output_features, self.inference_stage
        ).generate_scripted_module()
        self.model_ts.save(model_path)

        model_artifact = TritonArtifact(
            model_name=self.full_model_name,
            model_version=self.model_version,
            platform="pytorch_libtorch",
            path=model_path,
            content_type="application/octet-stream",
            content_length=model_size(self.model_ts),
        )

        return model_artifact

    def save_config(self) -> TritonArtifact:
        """Save the Triton config.

        Return the appropriate artifact.
        """
        device = self.device
        if self.inference_stage != PREDICTOR:
            device = "cpu"
        self.config = TritonConfig(
            self.full_model_name,
            self.input_features,
            self.output_features,
            self.max_batch_size,
            self.max_queue_delay_microseconds,
            device,
            self.model_instance_count,
            self.inference_stage,
        )

        config_path = os.path.join(self.base_path, "config.pbtxt")
        with open(config_path, "w") as f:
            formatted_config = remove_empty_lines(self.config.get_model_config())
            f.write(formatted_config)

        config_artifact = TritonArtifact(
            model_name=self.full_model_name,
            model_version=self.model_version,
            platform="pytorch_libtorch",
            path=config_path,
            content_type="text/x-protobuf",
            content_length=os.path.getsize(config_path),
        )

        return config_artifact


@DeveloperAPI
@dataclass
class TritonEnsembleConfig:
    """Dataclass for creating and saving the Triton ensemble config."""

    # TritonMaster object for the preprocessor.
    triton_master_preprocessor: TritonMaster

    # TritonMaster object for the predictor.
    triton_master_predictor: TritonMaster

    # TritonMaster object for the postprocessor.
    triton_master_postprocessor: TritonMaster

    # Name of the model as specified by the caller of the triton_export function.
    model_name: str

    # Base output directory. Corresponds to the Triton model registry.
    output_path: str

    # Triton model version.
    model_version: Union[int, str]

    def __post_init__(self):
        self.ensemble_model_name = self.model_name
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

    def _get_ensemble_scheduling_step(self, triton_master: TritonMaster) -> str:
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
                    data_type=feature.type,
                    data_dims=", ".join(str(dim) for dim in feature.dimension),  # check correctness
                    reshape_spec="",
                )
            )
        return ",".join(spec)

    def get_config(self) -> str:
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

    def save_ensemble_config(self) -> TritonArtifact:
        config_path = os.path.join(self.base_path, "config.pbtxt")
        with open(config_path, "w") as f:
            formatted_config = remove_empty_lines(self.get_config())
            f.write(formatted_config)

        config_artifact = TritonArtifact(
            model_name=self.ensemble_model_name,
            model_version=self.model_version,
            platform="ensemble",
            path=config_path,
            content_type="text/x-protobuf",
            content_length=os.path.getsize(config_path),
        )

        return config_artifact

    def save_ensemble_dummy_model(self) -> TritonArtifact:
        """Scripts the model and saves it."""
        if isinstance(self.model_version, str) and self.model_version.isdigit():
            self.model_version = int(self.model_version)
        if not isinstance(self.model_version, int) or self.model_version < 1:
            raise ValueError("Model version has to be a non-zero positive integer")
        pass

        os.makedirs(os.path.join(self.base_path, str(self.model_version)), exist_ok=True)
        model_path = os.path.join(self.base_path, str(self.model_version), "model.txt")
        with open(model_path, "w") as f:
            f.write("no model for the ensemble")

        model_artifact = TritonArtifact(
            model_name=self.ensemble_model_name,
            model_version=self.model_version,
            platform="ensemble",
            path=model_path,
            content_type="text/plain",
            content_length=os.path.getsize(model_path),
        )

        return model_artifact


@DeveloperAPI
@dataclass
class TritonConfig:
    """Enables the creation and export of a Triton config."""

    # Name of the model. Must be the same as the directory where the config is saved.
    full_model_name: str

    # Input features of the model.
    input_features: List[TritonConfigFeature]

    # Output features of the model.
    output_features: List[TritonConfigFeature]

    # Max batch size of the model. (Triton config param).
    max_batch_size: int

    # Max time a request to the Triton model spends in the queue. (Triton config param).
    max_queue_delay_microseconds: int

    # One of "cpu", "cuda".
    device: str

    # Number of model instances on device.
    model_instance_count: int

    # One of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    inference_stage: str

    def _get_triton_spec(self, triton_features: List[TritonConfigFeature]) -> str:
        spec = []
        for feature in triton_features:
            spec.append(
                TRITON_SPEC.format(
                    key=feature.key,
                    data_type=feature.type,
                    data_dims=", ".join(str(dim) for dim in feature.dimension),  # check correctness
                    reshape_spec=self._get_reshape_spec(feature),
                )
            )
        return ",".join(spec)

    def _get_reshape_spec(self, feature) -> str:
        if feature.kind == INPUT and self.inference_stage == PREDICTOR:
            return FEATURE_RESHAPE_SPEC.format(reshape_dims=", ".join(str(dim) for dim in feature.dimension[1:]))
        return ""

    def _get_instance_spec(self) -> str:
        if self.device == "cpu":
            kind = "KIND_CPU"
        else:
            kind = "KIND_GPU"
        spec = INSTANCE_SPEC.format(count=self.model_instance_count, kind=kind)
        return spec

    def _get_dynamic_batching_spec(self) -> str:
        if self.inference_stage == PREDICTOR:
            return DYNAMIC_BATCHING_TEMPLATE.format(delay=self.max_queue_delay_microseconds)
        return ""

    def get_model_config(self) -> str:
        """Generate a Triton config for a model from the input and output features."""
        max_batch_size = self.max_batch_size
        if self.inference_stage != PREDICTOR:
            max_batch_size = 0

        config = TRITON_CONFIG_TEMPLATE.format(
            model_name=self.full_model_name,
            max_batch_size=max_batch_size,
            dynamic_batching_spec=self._get_dynamic_batching_spec(),
            input_spec=self._get_triton_spec(self.input_features),
            output_spec=self._get_triton_spec(self.output_features),
            instance_spec=self._get_instance_spec(),
        )
        return config


@DeveloperAPI
@dataclass
class TritonModel:
    """Enables the scripting and export of a model."""

    # The inference module.
    module: Union[_InferencePreprocessor, _InferencePredictor, _InferencePostprocessor]

    # Input features of the model.
    input_features: List[TritonConfigFeature]

    # Output features of the model.
    output_features: List[TritonConfigFeature]

    # One of PREPROCESSOR, PREDICTOR, POSTPROCESSOR.
    inference_stage: str

    def _get_dict_type_hint(self) -> str:
        return {
            PREPROCESSOR: "TorchscriptPreprocessingInput",
            PREDICTOR: "torch.Tensor",
            POSTPROCESSOR: "torch.Tensor",
        }[self.inference_stage]

    def _get_input_signature(self, triton_features: List[TritonConfigFeature]) -> str:
        elems = [
            f"{feature.wrapper_signature_name}: {feature._get_wrapper_signature_type()}" for feature in triton_features
        ]
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
            input_type=self._get_dict_type_hint(),
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


def get_device_types_and_counts(
    preprocessor_num_instances: int,
    predictor_device_type: str,
    predictor_num_instances: int,
    postprocessor_num_instances: int,
) -> Tuple[List[str], List[int]]:
    """Retrun device types and instance counts for each of the three inference modules."""
    if predictor_device_type not in ["cuda", "cpu"]:
        raise ValueError('Invalid predictor device type. Choose one of ["cpu", "cuda"].')
    elif predictor_device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Specified num_gpus > 0, but CUDA isn't available.")

    preprocessor_device_type = "cpu"
    postprocessor_device_type = "cpu"
    device_types = [preprocessor_device_type, predictor_device_type, postprocessor_device_type]
    device_counts = [preprocessor_num_instances, predictor_num_instances, postprocessor_num_instances]
    return device_types, device_counts


def get_inference_modules(model: LudwigModel, predictor_device_type: str) -> List[torch.jit.ScriptModule]:
    """Return the three inference modules."""
    inference_module = InferenceModule.from_ludwig_model(
        model.model, model.config, model.training_set_metadata, device=predictor_device_type
    )
    return [inference_module.preprocessor, inference_module.predictor, inference_module.postprocessor]


def get_example_input(
    model: LudwigModel, device_types: List[str], data_example: Union[None, pd.DataFrame]
) -> Dict[str, TorchscriptPreprocessingInput]:
    """Return an inference module-compatible input example.

    Generates a synthetic example if one is not provided.
    """
    config = model.config
    if data_example is None:
        features = config["input_features"] + config["output_features"]
        df = build_synthetic_dataset(dataset_size=1, features=features)
        data = [row for row in df]
        data_example = pd.DataFrame(data[1:], columns=data[0])
    return to_inference_module_input_from_dataframe(
        data_example.head(1), config, load_paths=True, device=device_types[0]
    )


def clean_up_synthetic_data():
    """Clean up synthetic example generated data for audio and image features."""
    shutil.rmtree("audio_files", ignore_errors=True)
    shutil.rmtree("image_files", ignore_errors=True)


@DeveloperAPI
def export_triton(
    model: LudwigModel,
    data_example: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = "model_repository",
    model_name: Optional[str] = "ludwig_model",
    model_version: Optional[Union[int, str]] = 1,
    preprocessor_num_instances: Optional[int] = 1,
    predictor_device_type: Optional[str] = "cpu",
    predictor_num_instances: Optional[int] = 1,
    postprocessor_num_instances: Optional[int] = 1,
    predictor_max_batch_size: Optional[int] = 64,
    max_queue_delay_microseconds: Optional[int] = 100,
) -> List[TritonArtifact]:
    """Exports a torchscript model to a output path that serves as a repository for Triton Inference Server.

    # Inputs
    :param model: (LudwigModel) A ludwig model.
    :param data_example: (pd.DataFrame) an example from the dataset.
        Used to get dimensions throughout the pipeline.
    :param output_path: (str) The output path for the model repository.
    :param model_name: (str) The optional model name.
    :param model_version: (Union[int,str]) The optional model verison.
    :param preprocessor_num_instances: (int) number of instances for the preprocessor (on CPU).
    :param predictor_device_type: (str) device type for the predictor to be deployed on. One of "cpu" or "cuda"
    :param predictor_num_instances: (int) number of instances for the predictor.
    :param postprocessor_num_instances: (int) number of instances for the postprocessor (on CPU).
    :param predictor_max_batch_size: (int) max_batch_size parameter for the predictor Triton config.
    :param max_queue_delay_microseconds: (int) max_queue_delay_microseconds for all Triton configs.

    # Return
    :return: (List[TritonArtifact]) list of TritonArtifacts that contains information about exported artifacts.
    """

    device_types, instance_counts = get_device_types_and_counts(
        preprocessor_num_instances, predictor_device_type, predictor_num_instances, postprocessor_num_instances
    )
    split_modules = get_inference_modules(model, device_types[1])
    example_input = get_example_input(model, device_types, data_example)

    triton_masters = []
    triton_artifacts = []
    for i, module in enumerate(split_modules):
        example_input = place_on_device(example_input, device_types[i])
        triton_master = TritonMaster(
            module=module,
            input_data_example=example_input,
            inference_stage=INFERENCE_STAGES[i],
            max_batch_size=predictor_max_batch_size,
            max_queue_delay_microseconds=max_queue_delay_microseconds,
            model_name=model_name,
            output_path=output_path,
            model_version=model_version,
            ludwig_config=model.config,
            device=device_types[i],
            model_instance_count=instance_counts[i],
        )
        example_input = triton_master.output_data_example
        config_artifact = triton_master.save_config()
        model_artifact = triton_master.save_model()
        triton_masters.append(triton_master)
        triton_artifacts.extend([config_artifact, model_artifact])

    # saving ensemble config
    triton_master_preprocessor, triton_master_predictor, triton_master_postprocessor = triton_masters
    ensemble_config = TritonEnsembleConfig(
        triton_master_preprocessor,
        triton_master_predictor,
        triton_master_postprocessor,
        model_name,
        output_path,
        model_version,
    )
    config_artifact = ensemble_config.save_ensemble_config()
    model_artifact = ensemble_config.save_ensemble_dummy_model()
    triton_artifacts.extend([config_artifact, model_artifact])
    clean_up_synthetic_data()

    return triton_artifacts
