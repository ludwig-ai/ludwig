import os
from typing import Any, Dict, List, TYPE_CHECKING, Union

import pandas as pd
import torch
from torch import nn

from ludwig.constants import COLUMN, NAME, TYPE
from ludwig.data.postprocessing import convert_dict_to_df
from ludwig.data.preprocessing import load_metadata
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.features.feature_utils import get_module_dict_key_from_name, get_name_from_module_dict_key
from ludwig.globals import INFERENCE_MODULE_FILE_NAME, MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils import image_utils, output_feature_utils
from ludwig.utils.audio_utils import read_audio_if_path
from ludwig.utils.data_utils import load_json
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.types import TorchscriptPreprocessingInput

# Prevents circular import errors from typing.
if TYPE_CHECKING:
    from ludwig.models.ecd import ECD

INFERENCE_PREPROCESSOR_FILENAME = "inference_preprocessor.pt"
INFERENCE_PREDICTOR_FILENAME = "inference_predictor.pt"
INFERENCE_POSTPROCESSOR_FILENAME = "inference_postprocessor.pt"


class InferenceModule(nn.Module):
    """A nn.Module subclass that wraps the inference preprocessor, predictor, and postprocessor.

    Useful if deploying the model in a pure PyTorch backend.
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

    def forward(self, inputs: TorchscriptPreprocessingInput) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            preproc_outputs = self.preprocessor(inputs)
            predictions_flattened = self.predictor(preproc_outputs)
            postproc_outputs_flattened = self.postprocessor(predictions_flattened)
            # Turn flat inputs into nested predictions per feature name
            postproc_outputs = unflatten_dict_by_feature_name(postproc_outputs_flattened)
            return postproc_outputs


class InferencePreprocessor(nn.Module):
    """Wraps preprocessing modules into a single nn.Module.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        super().__init__()
        input_features = {
            feature[NAME]: get_from_registry(feature[TYPE], input_type_registry) for feature in config["input_features"]
        }
        self.preproc_modules = nn.ModuleDict()
        for feature_name, feature in input_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)  # prevents collisions with reserved keywords
            self.preproc_modules[module_dict_key] = feature.create_preproc_module(training_set_metadata[feature_name])

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            preproc_inputs: Dict[str, torch.Tensor] = {}
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

    def __init__(self, model: "ECD"):
        super().__init__()
        self.model = model.cpu().to_torchscript()
        self.predict_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)  # prevents collisions with reserved keywords
            self.predict_modules[module_dict_key] = feature.prediction_module

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

    def __init__(self, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        output_features = {
            feature[NAME]: get_from_registry(feature[TYPE], output_type_registry)
            for feature in config["output_features"]
        }
        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in output_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)  # prevents collisions with reserved keywords
            self.postproc_modules[module_dict_key] = feature.create_postproc_module(training_set_metadata[feature_name])

    def forward(self, predictions_flattened: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            # Turn flat inputs into nested predictions per feature name
            predictions = unflatten_dict_by_feature_name(predictions_flattened)
            postproc_outputs_flattened: Dict[str, Any] = {}
            for module_dict_key, postproc in self.postproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                feature_postproc_outputs = postproc(predictions[feature_name])
                # Flatten out the predictions to support Triton input/output
                for postproc_key, tensor_values in feature_postproc_outputs.items():
                    postproc_concat_key = output_feature_utils.get_feature_concat_name(feature_name, postproc_key)
                    postproc_outputs_flattened[postproc_concat_key] = tensor_values
            return postproc_outputs_flattened


def save_ludwig_model_for_inference(
    save_path: str, model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any]
) -> None:
    """Saves LudwigModel (represented by model, config, and training_set_metadata) for inference."""
    # Preprocessing modules
    preprocessor = torch.jit.script(InferencePreprocessor(config, training_set_metadata))
    preprocessor.save(os.path.join(save_path, INFERENCE_PREPROCESSOR_FILENAME))
    # Model forward pass modules
    predictor = torch.jit.script(InferencePredictor(model))
    predictor.save(os.path.join(save_path, INFERENCE_PREDICTOR_FILENAME))
    # Postprocessing modules
    postprocessor = torch.jit.script(InferencePostprocessor(config, training_set_metadata))
    postprocessor.save(os.path.join(save_path, INFERENCE_POSTPROCESSOR_FILENAME))


def init_inference_module_from_ludwig_model(
    model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any]
) -> InferenceModule:
    preprocessor = torch.jit.script(InferencePreprocessor(config, training_set_metadata))
    predictor = torch.jit.script(InferencePredictor(model))
    postprocessor = torch.jit.script(InferencePostprocessor(config, training_set_metadata))
    return InferenceModule(preprocessor, predictor, postprocessor)


def init_inference_module_from_directory(directory: str) -> InferenceModule:
    preprocessor = torch.jit.load(os.path.join(directory, INFERENCE_PREPROCESSOR_FILENAME))
    predictor = torch.jit.load(os.path.join(directory, INFERENCE_PREDICTOR_FILENAME))
    postprocessor = torch.jit.load(os.path.join(directory, INFERENCE_POSTPROCESSOR_FILENAME))
    return InferenceModule(preprocessor, predictor, postprocessor)


def unflatten_dict_by_feature_name(postproc_outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a flattened dictionary of tensors to a nested dictionary of outputs per feature name."""
    outputs: Dict[str, Dict[str, Any]] = {}
    for postproc_key, tensor_values in postproc_outputs.items():
        feature_name = output_feature_utils.get_feature_name_from_concat_name(postproc_key)
        tensor_name = output_feature_utils.get_tensor_name_from_concat_name(postproc_key)
        if feature_name not in outputs:
            outputs[feature_name] = {}
        outputs[feature_name][tensor_name] = tensor_values
    return outputs


class InferenceLudwigModel:
    """Model for inference with the subset of the LudwigModel interface used for prediction.

    This model is instantiated with a model_dir, which contains the model and its metadata.
    """

    def __init__(self, model_dir: str):
        self.model = torch.jit.load(os.path.join(model_dir, INFERENCE_MODULE_FILE_NAME))
        self.config = load_json(os.path.join(model_dir, MODEL_HYPERPARAMETERS_FILE_NAME))
        self.training_set_metadata = load_metadata(os.path.join(model_dir, TRAIN_SET_METADATA_FILE_NAME))

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

        preds = self.model(inputs)

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
