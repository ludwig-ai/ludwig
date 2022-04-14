from typing import Any, Dict, List, Union

import torch
from torch import nn

from ludwig.constants import NAME, TYPE
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.ecd import ECD
from ludwig.utils.misc_utils import get_from_registry


class InferenceModule(nn.Module):
    """Wraps preprocessing, model forward pass, and postprocessing into a single module.

    The purpose of the module is to be scripted into Torchscript for native serving.
    """

    def __init__(self, model: ECD, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        super().__init__()

        model.cpu()
        self.model = model.to_torchscript()

        input_features = {
            feature[NAME]: get_from_registry(feature[TYPE], input_type_registry) for feature in config["input_features"]
        }
        self.preproc_modules = nn.ModuleDict(
            {
                feature_name: feature.create_preproc_module(training_set_metadata[feature_name])
                for feature_name, feature in input_features.items()
            }
        )

        output_features = {
            feature[NAME]: get_from_registry(feature[TYPE], output_type_registry)
            for feature in config["output_features"]
        }
        self.predict_modules = nn.ModuleDict(
            {feature_name: feature.prediction_module for feature_name, feature in model.output_features.items()}
        )
        self.postproc_modules = nn.ModuleDict(
            {
                feature_name: feature.create_postproc_module(training_set_metadata[feature_name])
                for feature_name, feature in output_features.items()
            }
        )

    def forward(self, inputs: Dict[str, Union[List[str], List[torch.Tensor], torch.Tensor]]):
        with torch.no_grad():
            preproc_inputs = {
                feature_name: preproc(inputs[feature_name]) for feature_name, preproc in self.preproc_modules.items()
            }

            outputs = self.model(preproc_inputs)

            predictions = {
                feature_name: predict(outputs, feature_name) for feature_name, predict in self.predict_modules.items()
            }

            postproc_outputs = {
                feature_name: postproc(predictions[feature_name])
                for feature_name, postproc in self.postproc_modules.items()
            }

            return postproc_outputs
