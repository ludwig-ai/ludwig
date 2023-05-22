import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
import torch

from ludwig.combiners.combiners import create_combiner
from ludwig.constants import MODEL_ECD
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.model_types.ecd import ECDModelConfig
from ludwig.utils import output_feature_utils
from ludwig.utils.augmentation_utils import AugmentationPipelines
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.fs_utils import open_file
from ludwig.utils.state_dict_backward_compatibility import update_state_dict
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


class ECD(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_ECD

    def __init__(
        self,
        config_obj: ECDModelConfig,
        random_seed=None,
        **_kwargs,
    ):
        self.config_obj = config_obj
        self._random_seed = random_seed

        super().__init__(random_seed=self._random_seed)

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Combiner ================
        logger.debug(f"Combiner {self.config_obj.combiner.type}")
        self.combiner = create_combiner(self.config_obj.combiner, input_features=self.input_features)

        # ================ Outputs ================
        self.output_features.update(
            self.build_outputs(output_feature_configs=self.config_obj.output_features, combiner=self.combiner)
        )

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
            encoder = self.input_features.get(input_feature_name)
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

    def unskip(self):
        for k in self.input_features.keys():
            self.input_features.set(k, self.input_features.get(k).unskip())

    def save(self, save_path):
        """Saves the model to the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        torch.save(self.state_dict(), weights_save_path)

    def load(self, save_path):
        """Loads the model from the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        device = torch.device(get_torch_device())
        with open_file(weights_save_path, "rb") as f:
            state_dict = torch.load(f, map_location=device)
            self.load_state_dict(update_state_dict(state_dict))

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (
            self.config_obj.input_features.to_list(),
            self.config_obj.combiner.to_dict(),
            self.config_obj.output_features.to_list(),
            self._random_seed,
        )

    def get_augmentation_pipelines(self) -> AugmentationPipelines:
        """Returns the augmentation pipeline for this model."""
        # dictionary to hold any augmentation pipeline
        augmentation_pipelines = {}

        # loop through all input features and add their augmentation pipeline to the dictionary
        for input_feature in self.config_obj.input_features:
            # if augmentation was specified for this input feature, add AugmentationPipeline to dictionary
            if input_feature.has_augmentation():
                # use input feature proc_column as key because that is what is used in the Batcher
                augmentation_pipelines[input_feature.proc_column] = self.input_features.get(
                    input_feature.name
                ).get_augmentation_pipeline()

        return AugmentationPipelines(augmentation_pipelines)
