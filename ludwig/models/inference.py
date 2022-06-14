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
from ludwig.utils import image_utils, output_feature_utils
from ludwig.utils.audio_utils import read_audio_if_path
from ludwig.utils.data_utils import load_json
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
INFERENCE_PREPROCESSOR_PREFIX = "inference_preprocessor"
INFERENCE_PREDICTOR_PREFIX = "inference_predictor"
INFERENCE_POSTPROCESSOR_PREFIX = "inference_postprocessor"


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
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            preproc_outputs: Dict[str, torch.Tensor] = self.preprocessor(inputs)
            predictions_flattened: Dict[str, torch.Tensor] = self.predictor(preproc_outputs)
            postproc_outputs_flattened: Dict[str, Any] = self.postprocessor(predictions_flattened)
            # Turn flat inputs into nested predictions per feature name
            postproc_outputs: Dict[str, Dict[str, Any]] = unflatten_dict_by_feature_name(postproc_outputs_flattened)
            return postproc_outputs


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

    predictor = torch.jit.script(InferencePredictor(model, stage_to_device[PREDICTOR]))
    predictor.save(os.path.join(save_path, stage_to_filenames[PREDICTOR]))
    if model_only:
        return

    preprocessor = torch.jit.script(InferencePreprocessor(config, training_set_metadata, stage_to_device[PREPROCESSOR]))
    preprocessor.save(os.path.join(save_path, stage_to_filenames[PREPROCESSOR]))
    postprocessor = torch.jit.script(
        InferencePostprocessor(model, training_set_metadata, stage_to_device[POSTPROCESSOR])
    )
    postprocessor.save(os.path.join(save_path, stage_to_filenames[POSTPROCESSOR]))


def init_inference_module_from_ludwig_model(
    model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any], device: TorchDevice
) -> InferenceModule:
    """Initializes an InferenceModule from a LudwigModel (an ECD model, config, and training_set_metadata)."""
    stage_to_device = get_stage_to_device_dict(device)
    preprocessor = torch.jit.script(
        InferencePreprocessor(config, training_set_metadata, stage_to_device[PREPROCESSOR]),
    )
    predictor = torch.jit.script(
        InferencePredictor(model, stage_to_device[PREDICTOR]),
    )
    postprocessor = torch.jit.script(
        InferencePostprocessor(model, training_set_metadata, stage_to_device[POSTPROCESSOR]),
    )
    return InferenceModule(preprocessor, predictor, postprocessor)


def init_inference_module_from_directory(
    directory: str, device: Optional[Union[Dict[str, TorchDevice], TorchDevice]] = None
) -> InferenceModule:
    """Initializes an InferenceModule from a directory containing saved preproc/predict/postproc modules."""
    stage_to_device = get_stage_to_device_dict(device)
    stage_to_filenames = {stage: get_filename_from_stage(stage, device) for stage, device in stage_to_device.items()}

    preprocessor = torch.jit.load(os.path.join(directory, stage_to_filenames[PREPROCESSOR]))
    predictor = torch.jit.load(os.path.join(directory, stage_to_filenames[PREDICTOR]))
    postprocessor = torch.jit.load(os.path.join(directory, stage_to_filenames[POSTPROCESSOR]))
    return InferenceModule(preprocessor, predictor, postprocessor)


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
        if not list(device.keys()) == INFERENCE_STAGES:
            raise ValueError(f"Invalid device keys: {device}. Use {INFERENCE_STAGES}.")
        stage_to_device = {stage: torch.device(d) for stage, d in device.items()}
    else:
        raise ValueError(f"Invalid device: {device}.")
    return stage_to_device


def get_filename_from_stage(stage: str, device: Optional[TorchDevice] = None) -> str:
    """Returns the filename for a stage of inference."""
    if device is None:
        device = "cpu"
    if stage == PREPROCESSOR:
        return f"{INFERENCE_PREPROCESSOR_PREFIX}-{device}.pt"
    elif stage == PREDICTOR:
        return f"{INFERENCE_PREDICTOR_PREFIX}-{device}.pt"
    elif stage == POSTPROCESSOR:
        return f"{INFERENCE_POSTPROCESSOR_PREFIX}-{device}.pt"
    else:
        raise ValueError(f"Unknown stage: {stage}. Choose from: {INFERENCE_STAGES}.")


class InferenceLudwigModel:
    """Model for inference with the subset of the LudwigModel interface used for prediction.

    This model is instantiated with a model_dir, which contains the model and its metadata.

    This class is not intended to scriptable; instead, it is a wrapper around scripted modules with some additional
    functionality. If you need a fully scriptable module for end-to-end inference, use InferenceModule instead.

    Args:
        model_dir: Directory containing the model and its metadata.
        device: Device to use for inference. If None, use the default device. If `str` or `torch.device`, use the device
            specified. If `dict`, use the device specified for each of the preprocessor, predictor, and postprocessor.
    """

    def __init__(
        self, model_dir: str, device: Optional[Union[Dict[str, Union[str, torch.device]], str, torch.device]] = None
    ):
        self.stage_to_device = get_stage_to_device_dict(device)

        inference_module = init_inference_module_from_directory(model_dir, device=self.stage_to_device)
        self.preprocessor = inference_module.preprocessor
        self.predictor = inference_module.predictor
        self.postprocessor = inference_module.postprocessor

        self.config = load_json(os.path.join(model_dir, MODEL_HYPERPARAMETERS_FILE_NAME))
        # Do not remove; used in Predibase app
        self.training_set_metadata = load_metadata(os.path.join(model_dir, TRAIN_SET_METADATA_FILE_NAME))

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, Dict[str, Any]]:
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

    def predict(
        self, dataset: pd.DataFrame, return_type: Union[dict, pd.DataFrame] = pd.DataFrame
    ) -> Union[pd.DataFrame, dict]:
        """Predict on a batch of data.

        One difference between InferenceLudwigModel and LudwigModel is that the input data must be a pandas DataFrame.
        """
        inputs = {
            if_config["name"]: to_inference_module_input(dataset[if_config[COLUMN]], if_config[TYPE])
            for if_config in self.config["input_features"]
        }

        preds = self.forward(inputs)

        if return_type == pd.DataFrame:
            preds = convert_dict_to_df(preds)
        return preds, None  # Second return value is for compatibility with LudwigModel.predict


def to_inference_module_input(s: pd.Series, feature_type: str, load_paths=False) -> Union[List[str], torch.Tensor]:
    """Converts a pandas Series to be compatible with a torchscripted InferenceModule forward pass."""
    if feature_type == "image":
        if load_paths:
            return [image_utils.read_image(v) for v in s]
    elif feature_type == "audio":
        if load_paths:
            return [read_audio_if_path(v) for v in s]
    if feature_type in {"binary", "category", "bag", "set", "text", "sequence", "timeseries"}:
        return s.astype(str).to_list()
    return torch.from_numpy(s.to_numpy())
