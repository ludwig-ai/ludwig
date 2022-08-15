import copy
import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchmetrics

from ludwig.combiners.combiners import get_combiner_class
from ludwig.constants import MODEL_ECD, TYPE
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils import output_feature_utils
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


class ECD(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_ECD

    def __init__(
        self,
        input_features,
        combiner,
        output_features,
        random_seed=None,
        **_kwargs,
    ):
        self._input_features_def = copy.deepcopy(input_features)
        self._combiner_def = copy.deepcopy(combiner)
        self._output_features_def = copy.deepcopy(output_features)

        self._random_seed = random_seed

        super().__init__(random_seed=self._random_seed)

        # ================ Inputs ================
        self.input_features = LudwigFeatureDict()
        try:
            self.input_features.update(self.build_inputs(self._input_features_def))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Combiner ================
        logger.debug(f"Combiner {self._combiner_def[TYPE]}")
        combiner_class = get_combiner_class(self._combiner_def[TYPE])
        config, kwargs = load_config_with_kwargs(
            combiner_class.get_schema_cls(),
            self._combiner_def,
        )
        self.combiner = combiner_class(input_features=self.input_features, config=config, **kwargs)

        # ================ Outputs ================
        self.output_features = LudwigFeatureDict()
        self.output_features.update(self.build_outputs(self._output_features_def, self.combiner))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        # After constructing all layers, clear the cache to free up memory
        clear_data_cache()

    def encode(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ):
        # Convert inputs to tensors.
        for input_feature_name, input_values in inputs.items():
            if not isinstance(input_values, torch.Tensor):
                inputs[input_feature_name] = torch.from_numpy(input_values)
            else:
                inputs[input_feature_name] = input_values

        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs[input_feature_name] = encoder_output

        return encoder_outputs

    def combine(self, encoder_outputs):
        return self.combiner(encoder_outputs)

    def decode(self, combiner_outputs, targets, mask):
        # Invoke output features.
        output_logits = {}
        output_last_hidden = {}
        for output_feature_name, output_feature in self.output_features.items():
            # Use the presence or absence of targets to signal training or prediction.
            target = targets[output_feature_name] if targets is not None else None
            decoder_outputs = output_feature(combiner_outputs, output_last_hidden, mask=mask, target=target)

            # Add decoder outputs to overall output dictionary.
            for decoder_output_name, tensor in decoder_outputs.items():
                output_feature_utils.set_output_feature_tensor(
                    output_logits, output_feature_name, decoder_output_name, tensor
                )

            # Save the hidden state of the output feature (for feature dependencies).
            output_last_hidden[output_feature_name] = decoder_outputs["last_hidden"]
        return output_logits

    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs: Inputs to the model. Can be a dictionary of input names to
                input tensors or a tuple of (inputs, targets) where inputs is
                a dictionary of input names to input tensors and targets is a
                dictionary of target names to target tensors.
            mask: A mask for the inputs.

        Returns:
            A dictionary of output {feature name}::{tensor_name} -> output tensor.
        """

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

        encoder_outputs = self.encode(inputs)
        combiner_outputs = self.combine(encoder_outputs)
        return self.decode(combiner_outputs, targets, mask)

    def save(self, save_path):
        """Saves the model to the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        torch.save(self.state_dict(), weights_save_path)

    def load(self, save_path):
        """Loads the model from the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        device = torch.device(get_torch_device())
        self.load_state_dict(torch.load(weights_save_path, map_location=device))

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (self._input_features_df, self._combiner_def, self._output_features_df, self._random_seed)
