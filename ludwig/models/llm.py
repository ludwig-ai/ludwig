import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchmetrics

from ludwig.constants import MODEL_LLM
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.model_types.llm import LLMModelConfig
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.fs_utils import open_file
from ludwig.utils.state_dict_backward_compatibility import update_state_dict
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


class LLM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_LLM

    def __init__(
        self,
        config_obj: LLMModelConfig,
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

        # ================ Outputs ================
        self.output_features.update(
            self.build_outputs(output_feature_configs=self.config_obj.output_features, combiner=self.combiner)
        )

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        # After constructing all layers, clear the cache to free up memory
        clear_data_cache()

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

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=self.generation_config, max_new_tokens=4)
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return self.extract(outputs)

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
            self.config_obj.output_features.to_list(),
            self._random_seed,
        )
