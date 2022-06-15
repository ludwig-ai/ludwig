import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import torch
from torch import nn

from ludwig.constants import COLUMN, NAME, TYPE
from ludwig.data.postprocessing import convert_dict_to_df
from ludwig.data.preprocessing import load_metadata
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.features.feature_utils import get_module_dict_key_from_name, get_name_from_module_dict_key
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils import output_feature_utils
from ludwig.utils.audio_utils import read_audio_from_path
from ludwig.utils.data_utils import load_json
from ludwig.utils.image_utils import read_image_from_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import DEVICE, place_on_torch_device
from ludwig.utils.types import TorchDevice, TorchscriptPreprocessingInput

# Prevents circular import errors from typing.
if TYPE_CHECKING:
    from ludwig.models.ecd import ECD


PREPROCESSOR = "preprocessor"
PREDICTOR = "predictor"
POSTPROCESSOR = "postprocessor"
INFERENCE_STAGES = [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]


class _InferenceModuleV0(nn.Module):
    """Wraps preprocessing, model forward pass, and postprocessing into a single module.

    Deprecated; used for benchmarking against the new inference module.
    """

    def __init__(
        self, model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any], device: TorchDevice
    ):
        super().__init__()

        self.model = model.to_torchscript(device=device)

        input_features = {
            feature[NAME]: get_from_registry(feature[TYPE], input_type_registry) for feature in config["input_features"]
        }
        self.preproc_modules = nn.ModuleDict()
        for feature_name, feature in input_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.preproc_modules[module_dict_key] = feature.create_preproc_module(training_set_metadata[feature_name])

        self.predict_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.predict_modules[module_dict_key] = feature.prediction_module

        output_features = {
            feature[NAME]: get_from_registry(feature[TYPE], output_type_registry)
            for feature in config["output_features"]
        }
        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in output_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.postproc_modules[module_dict_key] = feature.create_postproc_module(training_set_metadata[feature_name])

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]):
        with torch.no_grad():
            preproc_inputs = {}
            for module_dict_key, preproc in self.preproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                preproc_inputs[feature_name] = preproc(inputs[feature_name])
            outputs = self.model(preproc_inputs)

            predictions_flattened: Dict[str, torch.Tensor] = {}
            for module_dict_key, predict in self.predict_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                feature_predictions = predict(outputs, feature_name)
                # Flatten out the predictions to support Triton input/output
                for predict_key, tensor_values in feature_predictions.items():
                    predict_concat_key = output_feature_utils.get_feature_concat_name(feature_name, predict_key)
                    predictions_flattened[predict_concat_key] = tensor_values

            postproc_outputs: Dict[str, Dict[str, Any]] = {}
            for module_dict_key, postproc in self.postproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                postproc_outputs[feature_name] = postproc(predictions_flattened, feature_name)

            return postproc_outputs


class InferenceModule(nn.Module):
    """A nn.Module subclass that wraps the inference preprocessor, predictor, and postprocessor.

    Note that if torch.jit.script is called on this class, all modules are packaged into a monolithic object. This is
    useful if deploying the model in a pure PyTorch/C++ backend. However, there are limitations, including the inability
    to place modules on different devices. If this functionality is required, use InferenceLudwigModel instead.
    """

    def __init__(
        self,
        preprocessor: torch.jit.ScriptModule,
        predictor: torch.jit.ScriptModule,
        postprocessor: torch.jit.ScriptModule,
        config: Optional[Dict[str, Any]] = None,
        training_set_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.config = config
        # Do not remove – used by Predibase app
        self.training_set_metadata = training_set_metadata

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            preproc_outputs: Dict[str, torch.Tensor] = self.preprocessor(inputs)
            predictions_flattened: Dict[str, torch.Tensor] = self.predictor(preproc_outputs)
            postproc_outputs_flattened: Dict[str, Any] = self.postprocessor(predictions_flattened)
            # Turn flat inputs into nested predictions per feature name
            postproc_outputs: Dict[str, Dict[str, Any]] = unflatten_dict_by_feature_name(postproc_outputs_flattened)
            return postproc_outputs

    @torch.jit.unused
    def forward_with_device_placement(
        self, inputs: Dict[str, TorchscriptPreprocessingInput]
    ) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            inputs = place_on_torch_device(inputs, self.preprocessor.device)
            preproc_outputs: Dict[str, torch.Tensor] = self.preprocessor(inputs)

            preproc_outputs = place_on_torch_device(preproc_outputs, self.predictor.device)
            predictions_flattened: Dict[str, torch.Tensor] = self.predictor(preproc_outputs)

            predictions_flattened = place_on_torch_device(predictions_flattened, self.postprocessor.device)
            postproc_outputs_flattened: Dict[str, Any] = self.postprocessor(predictions_flattened)

            # Turn flat inputs into nested predictions per feature name
            postproc_outputs: Dict[str, Dict[str, Any]] = unflatten_dict_by_feature_name(postproc_outputs_flattened)
            return postproc_outputs

    @torch.jit.unused
    def predict(
        self, dataset: pd.DataFrame, return_type: Union[dict, pd.DataFrame] = pd.DataFrame
    ) -> Union[pd.DataFrame, dict]:
        """Predict on a batch of data with an interface similar to LudwigModel.predict.

        One difference between InferenceLudwigModel and LudwigModel is that the input data must be a pandas DataFrame.
        """
        inputs = {
            if_config["name"]: to_inference_module_input(dataset[if_config[COLUMN]], if_config[TYPE])
            for if_config in self.config["input_features"]
        }

        preds = self.forward_with_device_placement(inputs)

        if return_type == pd.DataFrame:
            preds = convert_dict_to_df(preds)
        return preds, None  # Second return value is for compatibility with LudwigModel.predict

    @torch.jit.unused
    @classmethod
    def from_ludwig_model(
        cls: "InferenceModule",
        model: "ECD",
        config: Dict[str, Any],
        training_set_metadata: Dict[str, Any],
        device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None,
    ):
        stage_to_module = init_inference_stages_from_ludwig_model(
            model, config, training_set_metadata, device=device, scripted=True
        )

        return cls(
            stage_to_module[PREPROCESSOR],
            stage_to_module[PREDICTOR],
            stage_to_module[POSTPROCESSOR],
            config=config,
            training_set_metadata=training_set_metadata,
        )

    @torch.jit.unused
    @classmethod
    def from_directory(
        cls: "InferenceModule",
        directory: str,
        device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None,
    ):
        stage_to_module = init_inference_stages_from_directory(directory, device=device)

        config_path = os.path.join(directory, MODEL_HYPERPARAMETERS_FILE_NAME)
        config = load_json(config_path) if os.path.exists(config_path) else None

        metadata_path = os.path.join(directory, TRAIN_SET_METADATA_FILE_NAME)
        training_set_metadata = load_metadata(metadata_path) if os.path.exists(metadata_path) else None

        return cls(
            stage_to_module[PREPROCESSOR],
            stage_to_module[PREDICTOR],
            stage_to_module[POSTPROCESSOR],
            config=config,
            training_set_metadata=training_set_metadata,
        )


class InferencePreprocessor(nn.Module):
    """Wraps preprocessing modules into a single nn.Module.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, config: Dict[str, Any], training_set_metadata: Dict[str, Any], device: TorchDevice):
        super().__init__()
        self.device = torch.device(device)
        self.preproc_modules = nn.ModuleDict()
        for feature_config in config["input_features"]:
            feature_name = feature_config[NAME]
            feature = get_from_registry(feature_config[TYPE], input_type_registry)
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            feature_preproc_module = feature.create_preproc_module(training_set_metadata[feature_name])
            self.preproc_modules[module_dict_key] = feature_preproc_module.to(device=self.device)

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            preproc_inputs = {}
            for module_dict_key, preproc in self.preproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                preproc_inputs[feature_name] = preproc(inputs[feature_name])
            return preproc_inputs


class InferencePredictor(nn.Module):
    """Wraps model forward pass + predictions into a single nn.Module.

    The forward call of this module returns a flattened dictionary in order to support Triton input/output.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, model: "ECD", device: TorchDevice):
        super().__init__()
        self.device = torch.device(device)
        self.model = model.to_torchscript(self.device)
        self.predict_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.predict_modules[module_dict_key] = feature.prediction_module.to(device=self.device)

    def forward(self, preproc_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            model_outputs = self.model(preproc_inputs)
            predictions_flattened: Dict[str, torch.Tensor] = {}
            for module_dict_key, predict in self.predict_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                feature_predictions = predict(model_outputs, feature_name)
                # Flatten out the predictions to support Triton input/output
                for predict_key, tensor_values in feature_predictions.items():
                    predict_concat_key = output_feature_utils.get_feature_concat_name(feature_name, predict_key)
                    predictions_flattened[predict_concat_key] = tensor_values
            return predictions_flattened


class InferencePostprocessor(nn.Module):
    """Wraps postprocessing modules into a single nn.Module.

    The forward call of this module returns a flattened dictionary in order to support Triton input/output.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, model: "ECD", training_set_metadata: Dict[str, Any], device: TorchDevice):
        super().__init__()
        self.device = torch.device(device)
        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            feature_postproc_module = feature.create_postproc_module(training_set_metadata[feature_name])
            self.postproc_modules[module_dict_key] = feature_postproc_module.to(self.device)

    def forward(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        with torch.no_grad():
            postproc_outputs_flattened: Dict[str, Any] = {}
            for module_dict_key, postproc in self.postproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                feature_postproc_outputs = postproc(model_outputs, feature_name)
                # Flatten out the predictions to support Triton input/output
                for postproc_key, tensor_values in feature_postproc_outputs.items():
                    postproc_concat_key = output_feature_utils.get_feature_concat_name(feature_name, postproc_key)
                    postproc_outputs_flattened[postproc_concat_key] = tensor_values
            return postproc_outputs_flattened


def save_ludwig_model_for_inference(
    save_path: str,
    model: "ECD",
    config: Dict[str, Any],
    training_set_metadata: Dict[str, Any],
    model_only: bool = False,
    device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None,
) -> None:
    """Saves a LudwigModel (an ECD model, config, and training_set_metadata) for inference."""
    stage_to_device = get_stage_to_device_dict(device)
    stage_to_filenames = {stage: get_filename_from_stage(stage, device) for stage, device in stage_to_device.items()}

    stage_to_module = init_inference_stages_from_ludwig_model(
        model, config, training_set_metadata, device, scripted=True
    )
    if model_only:
        stage_to_module[PREDICTOR].save(os.path.join(save_path, stage_to_filenames[PREDICTOR]))
    else:
        for stage, module in stage_to_module.items():
            module.save(os.path.join(save_path, stage_to_filenames[stage]))


def init_inference_stages_from_directory(
    directory: str, device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None
) -> Dict[str, torch.nn.Module]:
    """Initializes inference stage modules from directory."""
    stage_to_device = get_stage_to_device_dict(device)
    stage_to_filenames = {stage: get_filename_from_stage(stage, device) for stage, device in stage_to_device.items()}

    stage_to_module = {}
    for stage in INFERENCE_STAGES:
        stage_to_module[stage] = torch.jit.load(
            os.path.join(directory, stage_to_filenames[stage]),
            map_location=stage_to_device[stage],
        )
    return stage_to_module


def init_inference_stages_from_ludwig_model(
    model: "ECD",
    config: Dict[str, Any],
    training_set_metadata: Dict[str, Any],
    device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None,
    scripted: bool = True,
) -> Dict[str, torch.nn.Module]:
    """Initializes inference stage modules from a LudwigModel (an ECD model, config, and training_set_metadata)."""
    stage_to_device = get_stage_to_device_dict(device)
    preprocessor = InferencePreprocessor(config, training_set_metadata, stage_to_device[PREPROCESSOR])
    predictor = InferencePredictor(model, stage_to_device[PREDICTOR])
    postprocessor = InferencePostprocessor(model, training_set_metadata, stage_to_device[POSTPROCESSOR])

    stage_to_module = {
        PREPROCESSOR: preprocessor,
        PREDICTOR: predictor,
        POSTPROCESSOR: postprocessor,
    }
    if scripted:
        stage_to_module = {stage: torch.jit.script(module) for stage, module in stage_to_module.items()}
    return stage_to_module


def to_inference_module_input(s: pd.Series, feature_type: str, load_paths=False) -> Union[List[str], torch.Tensor]:
    """Converts a pandas Series to be compatible with a torchscripted InferenceModule forward pass."""
    if feature_type == "image":
        if load_paths:
            return [read_image_from_path(v) if isinstance(v, str) else v for v in s]
    elif feature_type == "audio":
        if load_paths:
            return [read_audio_from_path(v) if isinstance(v, str) else v for v in s]
    if feature_type in {"binary", "category", "bag", "set", "text", "sequence", "timeseries"}:
        return s.astype(str).to_list()
    return torch.from_numpy(s.to_numpy())


def unflatten_dict_by_feature_name(flattened_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a flattened dictionary of objects to a nested dictionary of outputs per feature name."""
    outputs: Dict[str, Dict[str, Any]] = {}
    for concat_key, tensor_values in flattened_dict.items():
        feature_name = output_feature_utils.get_feature_name_from_concat_name(concat_key)
        tensor_name = output_feature_utils.get_tensor_name_from_concat_name(concat_key)
        feature_outputs: Dict[str, Any] = {}
        if feature_name not in outputs:
            outputs[feature_name] = feature_outputs
        else:
            feature_outputs = outputs[feature_name]
        feature_outputs[tensor_name] = tensor_values
    return outputs


def get_stage_to_device_dict(
    device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None
) -> Dict[str, torch.device]:
    """Returns a dict where each stage in INFERENCE_STAGES maps to a torch.device object."""
    stage_to_device = None
    if device is None:
        stage_to_device = {stage: torch.device(DEVICE) for stage in INFERENCE_STAGES}
    elif isinstance(device, str) or isinstance(device, torch.device):
        stage_to_device = {stage: torch.device(device) for stage in INFERENCE_STAGES}
    elif isinstance(device, dict):
        if not set(device.keys()) == set(INFERENCE_STAGES):
            raise ValueError(f"Invalid device keys: {device}. Use {INFERENCE_STAGES}.")
        stage_to_device = {stage: torch.device(d) for stage, d in device.items()}
    else:
        raise ValueError(f"Invalid device: {device}.")
    return stage_to_device


def get_filename_from_stage(stage: str, device: Optional[TorchDevice] = None) -> str:
    """Returns the filename for a stage of inference."""
    if device is None:
        device = "cpu"
    if stage not in INFERENCE_STAGES:
        raise ValueError(f"Invalid stage: {stage}.")
    return f"inference_{stage}-{device}.pt"
