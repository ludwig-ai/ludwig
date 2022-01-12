from typing import Any, Dict

from torch import nn

from ludwig.constants import NAME, TYPE
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.ecd import ECD
from ludwig.utils.misc_utils import get_from_registry


class InferenceModule(nn.Module):
    def __init__(self, model: ECD, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        self.model = model.to_torchscript()
        self.config = config
        self.training_set_metadata = training_set_metadata
        self.input_features = {
            feature[NAME]: get_from_registry(feature[TYPE], input_type_registry)(feature)
            for feature in self.config["input_features"]
        }

    def forward(self, inputs):
        preproc_inputs = {
            feature_name: feature.preprocess_inference_graph(
                inputs[feature_name], self.training_set_metadata[feature_name]
            )
            for feature_name, feature in self.input_features.items()
        }

        preproc_outputs = self.model.call(preproc_inputs)

        output_features = {
            feature[NAME]: get_from_registry(feature[TYPE], output_type_registry)(feature)
            for feature in self.config["output_features"]
        }

        preproc_preds = {
            feature_name: feature.predictions(preproc_outputs[feature_name], training=False)
            for feature_name, feature in output_features.items()
        }

        outputs = {
            feature_name: feature.postprocess_inference_graph(
                preproc_preds[feature_name], self.training_set_metadata[feature_name]
            )
            for feature_name, feature in output_features.items()
        }

        return outputs

    def sample_inputs(self):
        inputs = {
            feature_name: feature.create_inference_input(feature_name)
            for feature_name, feature in self.input_features.items()
        }

        return inputs
