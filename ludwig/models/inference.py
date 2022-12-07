import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import pandas as pd
import torch
from torch import nn

from ludwig.constants import NAME, POSTPROCESSOR, PREDICTOR, PREPROCESSOR, TYPE
from ludwig.data.postprocessing import convert_dict_to_df
from ludwig.data.preprocessing import load_metadata
from ludwig.features.feature_registries import get_input_type_registry
from ludwig.features.feature_utils import get_module_dict_key_from_name, get_name_from_module_dict_key
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.types import ModelConfigDict, TrainingSetMetadataDict
from ludwig.utils import output_feature_utils
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.inference_utils import get_filename_from_stage, to_inference_module_input_from_dataframe
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.output_feature_utils import get_feature_name_from_concat_name, get_tensor_name_from_concat_name
from ludwig.utils.torch_utils import DEVICE
from ludwig.utils.types import TorchDevice, TorchscriptPreprocessingInput

# Prevents circular import errors from typing.
if TYPE_CHECKING:
    from ludwig.models.base import BaseModel

logger = logging.getLogger(__name__)


class InferenceModule(nn.Module):
    """A nn.Module subclass that wraps the inference preprocessor, predictor, and postprocessor."""

    def __init__(
        self,
        preprocessor: torch.jit.ScriptModule,
        predictor: torch.jit.ScriptModule,
        postprocessor: torch.jit.ScriptModule,
        config: Optional[ModelConfigDict] = None,
        training_set_metadata: Optional[TrainingSetMetadataDict] = None,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.config = config
        # Do not remove – used by Predibase app
        self.training_set_metadata = training_set_metadata

    def preprocessor_forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, torch.Tensor]:
        """Forward pass through the preprocessor."""
        return self.preprocessor(inputs)

    def predictor_forward(self, preproc_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the predictor.

        Ensures that the inputs are on the correct device. The outputs are on the same device as self.predictor.
        """
        for k, v in preproc_inputs.items():
            preproc_inputs[k] = v.to(self.predictor.device)

        with torch.no_grad():  # Ensure model params do not compute gradients
            predictions_flattened = self.predictor(preproc_inputs)
            return predictions_flattened

    def postprocessor_forward(self, predictions_flattened: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Forward pass through the postprocessor."""
        postproc_outputs_flattened: Dict[str, Any] = self.postprocessor(predictions_flattened)
        # Turn flat inputs into nested predictions per feature name
        postproc_outputs: Dict[str, Dict[str, Any]] = _unflatten_dict_by_feature_name(postproc_outputs_flattened)
        return postproc_outputs

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, Dict[str, Any]]:
        preproc_inputs: Dict[str, torch.Tensor] = self.preprocessor_forward(inputs)
        predictions_flattened: Dict[str, torch.Tensor] = self.predictor_forward(preproc_inputs)
        postproc_outputs: Dict[str, Dict[str, Any]] = self.postprocessor_forward(predictions_flattened)
        return postproc_outputs

    @torch.jit.unused
    def predict(
        self, dataset: pd.DataFrame, return_type: Union[dict, pd.DataFrame] = pd.DataFrame
    ) -> Union[pd.DataFrame, dict]:
        """Predict on a batch of data with an interface similar to LudwigModel.predict."""
        inputs = to_inference_module_input_from_dataframe(dataset, self.config, load_paths=True)

        preds = self(inputs)

        if return_type == pd.DataFrame:
            preds = convert_dict_to_df(preds)
        return preds, None  # Second return value is for compatibility with LudwigModel.predict

    @torch.jit.unused
    @classmethod
    def from_ludwig_model(
        cls: "InferenceModule",
        model: "BaseModel",
        config: ModelConfigDict,
        training_set_metadata: TrainingSetMetadataDict,
        device: Optional[TorchDevice] = None,
    ):
        """Create an InferenceModule from a trained LudwigModel."""
        if device is None:
            logger.info(f'No device specified. Loading using device "{DEVICE}".')
            device = DEVICE

        stage_to_module = _init_inference_stages_from_ludwig_model(
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
        device: Optional[TorchDevice] = None,
    ):
        """Create an InferenceModule from a directory containing a model, config, and training set metadata."""
        if device is None:
            logger.info(f'No device specified. Loading using device "{DEVICE}".')
            device = DEVICE

        stage_to_module = _init_inference_stages_from_directory(directory, device=device)

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


class _InferencePreprocessor(nn.Module):
    """Wraps preprocessing modules into a single nn.Module.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, config: ModelConfigDict, training_set_metadata: TrainingSetMetadataDict):
        super().__init__()
        self.preproc_modules = nn.ModuleDict()
        for feature_config in config["input_features"]:
            feature_name = feature_config[NAME]
            feature = get_from_registry(feature_config[TYPE], get_input_type_registry())
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.preproc_modules[module_dict_key] = feature.create_preproc_module(training_set_metadata[feature_name])

    def forward(self, inputs: Dict[str, TorchscriptPreprocessingInput]) -> Dict[str, torch.Tensor]:
        preproc_inputs = {}
        for module_dict_key, preproc in self.preproc_modules.items():
            feature_name = get_name_from_module_dict_key(module_dict_key)
            preproc_inputs[feature_name] = preproc(inputs[feature_name])
        return preproc_inputs


class _InferencePredictor(nn.Module):
    """Wraps model forward pass + predictions into a single nn.Module.

    The forward call of this module returns a flattened dictionary in order to support Triton input/output.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, model: "BaseModel", device: TorchDevice):
        super().__init__()
        self.device = torch.device(device)
        self.model = model.to_torchscript(self.device)
        self.predict_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.predict_modules[module_dict_key] = feature.prediction_module.to(device=self.device)

    def forward(self, preproc_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


class _InferencePostprocessor(nn.Module):
    """Wraps postprocessing modules into a single nn.Module.

    The forward call of this module returns a flattened dictionary in order to support Triton input/output.

    TODO(geoffrey): Implement torchscript-compatible feature_utils.LudwigFeatureDict to replace
    get_module_dict_key_from_name and get_name_from_module_dict_key usage.
    """

    def __init__(self, model: "BaseModel", training_set_metadata: TrainingSetMetadataDict):
        super().__init__()
        self.postproc_modules = nn.ModuleDict()
        for feature_name, feature in model.output_features.items():
            # prevents collisions with reserved keywords
            module_dict_key = get_module_dict_key_from_name(feature_name)
            self.postproc_modules[module_dict_key] = feature.create_postproc_module(training_set_metadata[feature_name])

    def forward(self, predictions_flattened: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        postproc_outputs_flattened: Dict[str, Any] = {}
        for module_dict_key, postproc in self.postproc_modules.items():
            feature_name = get_name_from_module_dict_key(module_dict_key)
            feature_postproc_outputs = postproc(predictions_flattened, feature_name)
            # Flatten out the predictions to support Triton input/output
            for postproc_key, tensor_values in feature_postproc_outputs.items():
                postproc_concat_key = output_feature_utils.get_feature_concat_name(feature_name, postproc_key)
                postproc_outputs_flattened[postproc_concat_key] = tensor_values
        return postproc_outputs_flattened


def save_ludwig_model_for_inference(
    save_path: str,
    model: "BaseModel",
    config: ModelConfigDict,
    training_set_metadata: TrainingSetMetadataDict,
    device: Optional[TorchDevice] = None,
    model_only: bool = False,
) -> None:
    """Saves a LudwigModel (a BaseModel model, config, and training_set_metadata) for inference."""
    if device is None:
        logger.info(f'No device specified. Saving using device "{DEVICE}".')
        device = DEVICE

    stage_to_filenames = {
        stage: get_filename_from_stage(stage, device) for stage in [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]
    }

    stage_to_module = _init_inference_stages_from_ludwig_model(
        model, config, training_set_metadata, device, scripted=True
    )
    if model_only:
        stage_to_module[PREDICTOR].save(os.path.join(save_path, stage_to_filenames[PREDICTOR]))
    else:
        config_path = os.path.join(save_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        if not os.path.exists(config_path):
            save_json(config_path, config)
            logger.info(f"Saved model config to {config_path}")

        training_set_metadata_path = os.path.join(save_path, TRAIN_SET_METADATA_FILE_NAME)
        if not os.path.exists(training_set_metadata_path):
            save_json(training_set_metadata_path, training_set_metadata)
            logger.info(f"Saved training set metadata to {training_set_metadata_path}")

        for stage, module in stage_to_module.items():
            module.save(os.path.join(save_path, stage_to_filenames[stage]))
            logger.info(f"Saved torchscript module for {stage} to {stage_to_filenames[stage]}.")


def _init_inference_stages_from_directory(
    directory: str,
    device: TorchDevice,
) -> Dict[str, torch.nn.Module]:
    """Initializes inference stage modules from directory."""
    stage_to_filenames = {
        stage: get_filename_from_stage(stage, device) for stage in [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]
    }

    stage_to_module = {}
    for stage in [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]:
        stage_to_module[stage] = torch.jit.load(os.path.join(directory, stage_to_filenames[stage]))
        print(f"Loaded torchscript module for {stage} from {stage_to_filenames[stage]}.")
    return stage_to_module


def _init_inference_stages_from_ludwig_model(
    model: "BaseModel",
    config: ModelConfigDict,
    training_set_metadata: TrainingSetMetadataDict,
    device: TorchDevice,
    scripted: bool = True,
) -> Dict[str, torch.nn.Module]:
    """Initializes inference stage modules from a LudwigModel (a BaseModel model, config, and
    training_set_metadata)."""
    preprocessor = _InferencePreprocessor(config, training_set_metadata)
    predictor = _InferencePredictor(model, device=device)
    postprocessor = _InferencePostprocessor(model, training_set_metadata)

    stage_to_module = {
        PREPROCESSOR: preprocessor,
        PREDICTOR: predictor,
        POSTPROCESSOR: postprocessor,
    }
    if scripted:
        stage_to_module = {stage: torch.jit.script(module) for stage, module in stage_to_module.items()}
    return stage_to_module


def _unflatten_dict_by_feature_name(flattened_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a flattened dictionary of objects to a nested dictionary of outputs per feature name."""
    outputs: Dict[str, Dict[str, Any]] = {}
    for concat_key, tensor_values in flattened_dict.items():
        feature_name = get_feature_name_from_concat_name(concat_key)
        tensor_name = get_tensor_name_from_concat_name(concat_key)
        feature_outputs: Dict[str, Any] = {}
        if feature_name not in outputs:
            outputs[feature_name] = feature_outputs
        else:
            feature_outputs = outputs[feature_name]
        feature_outputs[tensor_name] = tensor_values
    return outputs
