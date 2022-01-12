from typing import Any, Dict

from torch import nn

from ludwig.constants import NAME, TYPE
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.ecd import ECD
from ludwig.utils.misc_utils import get_from_registry


class InferenceModule(nn.Module):
    def __init__(self, model: ECD, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        super().__init__()
        self.model = model.to_torchscript()

        input_features = {
            feature[NAME]: get_from_registry(feature[TYPE], input_type_registry)(feature)
            for feature in self.config["input_features"]
        }
        self.preproc_modules = nn.ModuleDict(
            {
                feature_name: feature.create_preproc_module(training_set_metadata)
                for feature_name, feature in input_features.items()
            }
        )

        output_features = {
            feature[NAME]: get_from_registry(feature[TYPE], output_type_registry)(feature)
            for feature in self.config["output_features"]
        }
        self.postproc_modules = nn.ModuleDict(
            {
                feature_name: feature.create_postproc_module(training_set_metadata)
                for feature_name, feature in output_features.items()
            }
        )

    def forward(self, inputs):
        preproc_inputs = {
            feature_name: self.preproc_modules[feature_name](inpt) for feature_name, inpt in inputs.items()
        }

        model_outputs = self.model(preproc_inputs)

        postproc_outputs = {
            feature_name: self.postproc_modules[feature_name](outpt) for feature_name, outpt in model_outputs.items()
        }

        return postproc_outputs
