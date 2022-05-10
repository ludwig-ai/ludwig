import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics

from ludwig.constants import LOGITS, NAME
from ludwig.features.base_feature import OutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.abstractmodel import AbstractModel
from ludwig.models.ecd import build_inputs, build_single_output
from ludwig.utils import output_feature_utils


def build_outputs(output_features_def: List[Dict[str, Any]], input_size: int) -> Dict[str, OutputFeature]:
    """Builds and returns output feature."""
    # TODO: only single task currently
    if len(output_features_def) > 1:
        raise ValueError("Only single task currently supported")

    output_feature_def = output_features_def[0]
    output_features = {}

    output_feature_def["input_size"] = input_size
    output_feature = build_single_output(output_feature_def, output_features)
    output_features[output_feature_def[NAME]] = output_feature

    return output_features


class GBM(AbstractModel):
    def __init__(self, input_features, output_features, random_seed=None, **_kwargs):
        self._input_features_def = copy.deepcopy(input_features)
        self._output_features_def = copy.deepcopy(output_features)

        super().__init__(random_seed=random_seed)

        # ================ Inputs ================
        self.input_features = LudwigFeatureDict()
        try:
            self.input_features.update(build_inputs(self._input_features_def))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Outputs ================
        self.output_features = LudwigFeatureDict()
        self.output_features.update(build_outputs(self._output_features_def, input_size=self.input_shape[-1]))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        self.compiled_model = None

    def set_compiled_model(self, model: nn.Module):
        self.compiled_model = model

    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        if self.compiled_model is None:
            raise ValueError("Model has not been trained yet.")

        if isinstance(inputs, tuple):
            inputs, targets = inputs
            # Convert targets to tensors.
            for target_feature_name, target_value in targets.items():
                if not isinstance(target_value, torch.Tensor):
                    targets[target_feature_name] = torch.from_numpy(target_value)
                else:
                    targets[target_feature_name] = target_value
        else:
            targets = None

        assert list(inputs.keys()) == self.input_features.keys()

        # Convert inputs to tensors.
        for input_feature_name, input_values in inputs.items():
            if not isinstance(input_values, torch.Tensor):
                inputs[input_feature_name] = torch.from_numpy(input_values)
            else:
                inputs[input_feature_name] = input_values.view(-1, 1)

        # concatenate inputs (TODO: concat combiner?)
        inputs = torch.cat(list(inputs.values()), dim=1)

        # Invoke output features.
        output_logits = {}
        # output_last_hidden = {}
        output_feature_name = self.output_features.keys()[0]
        # # Use the presence or absence of targets to signal training or prediction.
        # target = targets[output_feature_name] if targets is not None else None
        # # Add decoder outputs to overall output dictionary.
        # for decoder_output_name, tensor in decoder_outputs.items():
        output_feature_utils.set_output_feature_tensor(
            output_logits, output_feature_name, LOGITS, self.compiled_model(inputs)
        )
        # # Save the hidden state of the output feature (for feature dependencies).
        # output_last_hidden[output_feature_name] = decoder_outputs["last_hidden"]

        return output_logits

        # TODO(travis): include encoder and decoder steps during inference

        # encoder_outputs = {}
        # for input_feature_name, input_values in inputs.items():
        #     encoder = self.input_features[input_feature_name]
        #     encoder_output = encoder(input_values)
        #     encoder_outputs[input_feature_name] = encoder_output

        # combiner_outputs = self.combiner(encoder_outputs)
        #
