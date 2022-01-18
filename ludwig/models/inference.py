from typing import Any, Dict, List, Union

import torch
from torch import nn
from torch._C import ScriptModule

from ludwig.constants import NAME, TYPE
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.ecd import ECD
from ludwig.models.predictor import EXCLUDE_PRED_SET
from ludwig.utils.misc_utils import get_from_registry


class _ModelWithPreds(nn.Module):
    def __init__(self, model: ScriptModule, postproc_modules: nn.ModuleDict):
        super().__init__()
        self.model = model
        self.postproc_modules = postproc_modules

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Union[str, torch.Tensor]]:
        preds = self.model(inputs)
        print("PREDS", preds)

        postproc_outputs = {
            feature_name: postproc(preds[feature_name]) for feature_name, postproc in self.postproc_modules.items()
        }
        print("POSTPROC OUTPUTS", postproc_outputs)

        flat_tensor_outputs = {}
        flat_str_outputs = {}
        for of_name, of_preds in postproc_outputs.items():
            for pred_name, pred_values in of_preds.items():
                if pred_name not in EXCLUDE_PRED_SET:
                    key = f"{of_name}_{pred_name}"
                    if isinstance(pred_values, str):
                        flat_str_outputs[key] = pred_values
                    else:
                        flat_tensor_outputs[key] = pred_values
                    # flat_outputs[key] = pred_values
        print("FLAT TENSOR OUTPUTS", flat_tensor_outputs)
        print("FLAT STR OUTPUTS", flat_str_outputs)

        return flat_tensor_outputs, flat_str_outputs


class InferenceModule(nn.Module):
    def __init__(self, model: ECD, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        super().__init__()

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
            {
                feature_name: feature.create_predict_module(training_set_metadata[feature_name])
                for feature_name, feature in output_features.items()
            }
        )
        self.postproc_modules = nn.ModuleDict(
            {
                feature_name: feature.create_postproc_module(training_set_metadata[feature_name])
                for feature_name, feature in output_features.items()
            }
        )

        # model_with_preds = _ModelWithPreds(self.script_model, postproc_modules)
        # model_inputs = model.get_model_inputs()
        # self.model = torch.jit.script(model_with_preds)

    def forward(self, inputs: Dict[str, Union[List[str], torch.Tensor]]):
        with torch.no_grad():
            preproc_inputs = {
                feature_name: preproc(inputs[feature_name]) for feature_name, preproc in self.preproc_modules.items()
            }

            print(preproc_inputs)
            outputs = self.model(preproc_inputs)

            predictions = {
                feature_name: predict(outputs, feature_name) for feature_name, predict in self.predict_modules.items()
            }

            postproc_outputs = {
                feature_name: postproc(predictions[feature_name])
                for feature_name, postproc in self.postproc_modules.items()
            }

            return postproc_outputs
