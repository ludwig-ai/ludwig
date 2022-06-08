import os
from typing import Any, Dict, List, TYPE_CHECKING, Union

import pandas as pd
import torch
from torch import nn

from ludwig.constants import COLUMN, NAME, TYPE
from ludwig.data.postprocessing import convert_dict_to_df
from ludwig.data.preprocessing import load_metadata
from ludwig.features.feature_registries import input_type_registry
from ludwig.features.feature_utils import get_module_dict_key_from_name, get_name_from_module_dict_key
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils import image_utils, output_feature_utils
from ludwig.utils.audio_utils import read_audio_if_path
from ludwig.utils.data_utils import load_json
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.types import TorchscriptPreprocessingInput

# Prevents circular import errors from typing.
if TYPE_CHECKING:
    from ludwig.models.ecd import ECD

INFERENCE_PREPROCESSOR_FILENAME = "inference_preprocessor.pt"
INFERENCE_ECD_FILENAME = "inference_ecd.pt"
INFERENCE_POSTPROCESSOR_FILENAME = "inference_postprocessor.pt"


class _InferenceModuleV0(nn.Module):
    """Wraps preprocessing, model forward pass, and postprocessing into a single module.

    Deprecated; used for benchmarking against the new inference module.
    """

    def __init__(self, model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        super().__init__()

        self.model = model.cpu().to_torchscript()

        self.preproc_modules = nn.ModuleDict()
        for feature_config in config["input_features"]:
            feature_name = feature_config[NAME]
            feature = get_from_registry(feature_config[TYPE], input_type_registry)
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.preproc_modules[module_dict_key] = feature.create_preproc_module(training_set_metadata[feature_name])

        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.postproc_modules[module_dict_key] = feature.create_postproc_module(training_set_metadata[feature_name])

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]):
        with torch.no_grad():
            preproc_inputs = {}
            for module_dict_key, preproc in self.preproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                preproc_inputs[feature_name] = preproc(inputs[feature_name])
            outputs = self.model(preproc_inputs)

            postproc_outputs: Dict[str, Dict[str, Any]] = {}
            for module_dict_key, postproc in self.postproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                postproc_outputs[feature_name] = postproc(outputs, feature_name)

            return postproc_outputs


class InferenceModule(nn.Module):
    """A nn.Module subclass that wraps the inference preprocessor, predictor, and postprocessor.

    Useful if deploying the model in a pure PyTorch backend.
    """

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        preprocessor: torch.jit.ScriptModule,
        postprocessor: torch.jit.ScriptModule,
    ):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, Dict[str, Any]]:
        with torch.no_grad():
            preproc_outputs: Dict[str, torch.Tensor] = self.preprocessor(inputs)
            model_outputs: Dict[str, torch.Tensor] = self.model(preproc_outputs)
            postproc_outputs_flattened: Dict[str, Any] = self.postprocessor(model_outputs)
            # Turn flat inputs into nested predictions per feature name
            postproc_outputs: Dict[str, Dict[str, Any]] = unflatten_dict_by_feature_name(postproc_outputs_flattened)
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
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.preproc_modules[module_dict_key] = feature.create_preproc_module(training_set_metadata[feature_name])

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            preproc_inputs = {}
            for module_dict_key, preproc in self.preproc_modules.items():
                feature_name = get_name_from_module_dict_key(module_dict_key)
                preproc_inputs[feature_name] = preproc(inputs[feature_name])
            return preproc_inputs


class InferencePostprocessor(nn.Module):
    """Wraps postprocessing modules into a single nn.Module.

    The forward call of this module returns a flattened dictionary in order to support Triton input/output.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, model: "ECD", training_set_metadata: Dict[str, Any]):
        super().__init__()
        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.postproc_modules[module_dict_key] = feature.create_postproc_module(training_set_metadata[feature_name])

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
) -> None:
    """Saves a LudwigModel (an ECD model, config, and training_set_metadata) for inference."""
    scripted_model = model.cpu().to_torchscript()
    scripted_model.save(os.path.join(save_path, INFERENCE_ECD_FILENAME))
    if model_only:
        return
    preprocessor = torch.jit.script(InferencePreprocessor(config, training_set_metadata))
    preprocessor.save(os.path.join(save_path, INFERENCE_PREPROCESSOR_FILENAME))
    postprocessor = torch.jit.script(InferencePostprocessor(model, training_set_metadata))
    postprocessor.save(os.path.join(save_path, INFERENCE_POSTPROCESSOR_FILENAME))


def init_inference_module_from_ludwig_model(
    model: "ECD", config: Dict[str, Any], training_set_metadata: Dict[str, Any]
) -> InferenceModule:
    """Initializes an InferenceModule from a LudwigModel (an ECD model, config, and training_set_metadata)."""
    scripted_model = model.cpu().to_torchscript()
    preprocessor = torch.jit.script(InferencePreprocessor(config, training_set_metadata))
    postprocessor = torch.jit.script(InferencePostprocessor(model, training_set_metadata))
    return InferenceModule(scripted_model, preprocessor, postprocessor)


def init_inference_module_from_directory(directory: str) -> InferenceModule:
    """Initializes an InferenceModule from a directory containing saved preproc/predict/postproc modules."""
    scripted_model = torch.jit.load(os.path.join(directory, INFERENCE_ECD_FILENAME))
    preprocessor = torch.jit.load(os.path.join(directory, INFERENCE_PREPROCESSOR_FILENAME))
    postprocessor = torch.jit.load(os.path.join(directory, INFERENCE_POSTPROCESSOR_FILENAME))
    return InferenceModule(scripted_model, preprocessor, postprocessor)


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


class InferenceLudwigModel:
    """Model for inference with the subset of the LudwigModel interface used for prediction.

    This model is instantiated with a model_dir, which contains the model and its metadata.
    """

    def __init__(self, model_dir: str):
        self.model = init_inference_module_from_directory(model_dir)
        self.config = load_json(os.path.join(model_dir, MODEL_HYPERPARAMETERS_FILE_NAME))
        # Do not remove; used in the Predibase app
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
