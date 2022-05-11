import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ludwig.constants import COMBINED, LOSS
from ludwig.utils.metric_utils import get_scalar_from_ludwig_metric
from ludwig.utils.torch_utils import LudwigModule, reg_loss

logger = logging.getLogger(__name__)


class AbstractModel(LudwigModule):
    def __init__(self, random_seed=None):
        self._random_seed = random_seed

        # TODO: with change to misc_utils.set_random_seed() this may be redundant
        #       seems to be required for test_api.py::test_api_training_determinism
        if random_seed is not None:
            torch.random.manual_seed(random_seed)

        super().__init__()

    def get_model_inputs(self):
        inputs = {
            input_feature_name: input_feature.create_sample_input()
            for input_feature_name, input_feature in self.input_features.items()
        }
        return inputs

    # Return total number of parameters in model
    def get_model_size(self) -> int:
        model_tensors = self.collect_weights()
        total_size = 0
        for tnsr in model_tensors:
            total_size += tnsr[1].detach().cpu().numpy().size
        return total_size

    def to_torchscript(self):
        self.eval()
        model_inputs = self.get_model_inputs()
        # We set strict=False to enable dict inputs and outputs.
        return torch.jit.trace(self, model_inputs, strict=False)

    def save_torchscript(self, save_path):
        traced = self.to_torchscript()
        traced.save(save_path)

    @property
    def input_shape(self):
        # TODO(justin): Remove dummy implementation. Make input_shape and output_shape functions.
        return torch.Size([1, 1])

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
        raise NotImplementedError("Abstract method forward() should be implemented in subclass.")

    def predictions(self, inputs):
        outputs = self(inputs)
        predictions = {}
        for of_name in self.output_features:
            predictions[of_name] = self.output_features[of_name].predictions(outputs, of_name)
        return predictions

    def evaluation_step(self, inputs, targets):
        predictions = self.predictions(inputs)
        self.update_metrics(targets, predictions)
        return predictions

    def predict_step(self, inputs):
        return self.predictions(inputs)

    def train_loss(
        self,
        targets,
        predictions,
        regularization_type: Optional[str] = None,
        regularization_lambda: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the training loss for the model.

        Args:
            targets: A dictionary of target names to target tensors.
            predictions: A dictionary of output names to output tensors.
            regularization_type: One of 'l1', 'l2', 'l1_l2', or None.
            regularization_lambda: The regularization lambda.
        Returns:
            A tuple of the loss tensor and a dictionary of loss for every
            output feature.
        """
        train_loss = 0
        of_train_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_train_loss = of_obj.train_loss(targets[of_name], predictions, of_name)
            train_loss += of_obj.loss["weight"] * of_train_loss
            of_train_losses[of_name] = of_train_loss

        for loss in self.losses():
            train_loss += loss

        additional_losses = self.losses()
        if additional_losses:
            train_loss += torch.sum(torch.stack(additional_losses))  # other losses

        # Add regularization loss
        if regularization_type is not None and regularization_lambda != 0:
            train_loss += reg_loss(self, regularization_type, l1=regularization_lambda, l2=regularization_lambda)

        return train_loss, of_train_losses

    def eval_loss(self, targets, predictions):
        eval_loss = 0
        for of_name, of_obj in self.output_features.items():
            of_eval_loss = of_obj.eval_loss(targets[of_name], predictions[of_name])
            eval_loss += of_obj.loss["weight"] * of_eval_loss

        additional_loss = 0
        additional_losses = self.losses()
        if additional_losses:
            additional_loss = torch.sum(torch.stack(additional_losses))  # other losses

        return eval_loss, additional_loss

    def update_metrics(self, targets, predictions):
        for of_name, of_obj in self.output_features.items():
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        eval_loss, additional_losses = self.eval_loss(targets, predictions)
        self.eval_loss_metric.update(eval_loss)
        self.eval_additional_losses_metrics.update(additional_losses)

    def get_metrics(self):
        all_of_metrics = {}
        for of_name, of_obj in self.output_features.items():
            all_of_metrics[of_name] = of_obj.get_metrics()
        all_of_metrics[COMBINED] = {
            LOSS: get_scalar_from_ludwig_metric(self.eval_loss_metric)
            + get_scalar_from_ludwig_metric(self.eval_additional_losses_metrics)
        }
        return all_of_metrics

    def reset_metrics(self):
        for of_obj in self.output_features.values():
            of_obj.reset_metrics()
        self.eval_loss_metric.reset()

    def collect_weights(self, tensor_names=None, **kwargs):
        """Returns named parameters filtered against `tensor_names` if not None."""
        if not tensor_names:
            return self.named_parameters()

        # Check for bad tensor names.
        weight_names = {name for name, _ in self.named_parameters()}
        for name in tensor_names:
            if name not in weight_names:
                raise ValueError(f'Requested tensor name filter "{name}" not present in the model graph')

        # Apply filter.
        tensor_set = set(tensor_names)
        return [named_param for named_param in self.named_parameters() if named_param[0] in tensor_set]

    def get_args(self):
        return (self._input_features_df, self._combiner_def, self._output_features_df, self._random_seed)
