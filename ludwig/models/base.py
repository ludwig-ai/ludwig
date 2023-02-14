import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ludwig.combiners.combiners import Combiner
from ludwig.constants import COMBINED, LOSS, NAME
from ludwig.encoders.base import Encoder
from ludwig.features.base_feature import create_passthrough_input_feature, InputFeature, OutputFeature
from ludwig.features.feature_registries import get_input_type_registry, get_output_type_registry
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, FeatureCollection
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.metric_utils import get_scalar_from_ludwig_metric
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import DEVICE, LudwigModule, reg_loss
from ludwig.utils.types import TorchDevice

logger = logging.getLogger(__name__)


class BaseModel(LudwigModule, metaclass=ABCMeta):
    """Base model for use in LudwigModule.

    Implementations of this class should implement the following methods:
    - type()
    - forward()
    """

    @staticmethod
    @abstractmethod
    def type() -> str:
        """Returns the model type."""

    def __init__(self, random_seed: int = None):
        self._random_seed = random_seed

        # TODO: with change to misc_utils.set_random_seed() this may be redundant
        #       seems to be required for test_api.py::test_api_training_determinism
        if random_seed is not None:
            torch.random.manual_seed(random_seed)

        super().__init__()

        self.input_features = LudwigFeatureDict()
        self.output_features = LudwigFeatureDict()

    @classmethod
    def build_inputs(cls, input_feature_configs: FeatureCollection[BaseInputFeatureConfig]) -> Dict[str, InputFeature]:
        """Builds and returns input features in topological order."""
        input_features = OrderedDict()
        input_features_def = topological_sort_feature_dependencies(input_feature_configs.to_list())
        for input_feature_def in input_features_def:
            input_features[input_feature_def[NAME]] = cls.build_single_input(
                getattr(input_feature_configs, input_feature_def[NAME]), input_features
            )
        return input_features

    @staticmethod
    def build_single_input(
        feature_config: BaseInputFeatureConfig, other_input_features: Optional[Dict[str, InputFeature]]
    ) -> InputFeature:
        """Builds a single input feature from the input feature definition."""
        logger.debug(f"Input {feature_config.type} feature {feature_config.name}")

        encoder_obj = None
        if feature_config.tied is not None:
            tied_input_feature_name = feature_config.tied
            if tied_input_feature_name in other_input_features:
                encoder_obj = other_input_features[tied_input_feature_name].encoder_obj

        return create_input_feature(feature_config, encoder_obj)

    @classmethod
    def build_outputs(
        cls, output_feature_configs: FeatureCollection[BaseOutputFeatureConfig], combiner: Combiner
    ) -> Dict[str, OutputFeature]:
        """Builds and returns output features in topological order."""
        output_features_def = topological_sort_feature_dependencies(output_feature_configs.to_list())
        output_features = {}

        for output_feature_def in output_features_def:
            # TODO(Justin): Check that the semantics of input_size align with what the combiner's output shape returns
            # for seq2seq.
            setattr(getattr(output_feature_configs, output_feature_def[NAME]), "input_size", combiner.output_shape[-1])
            output_features[output_feature_def[NAME]] = cls.build_single_output(
                getattr(output_feature_configs, output_feature_def[NAME]), output_features
            )
        return output_features

    @staticmethod
    def build_single_output(
        feature_config: BaseOutputFeatureConfig, output_features: Optional[Dict[str, OutputFeature]]
    ) -> OutputFeature:
        """Builds a single output feature from the output feature definition."""
        logger.debug(f"Output {feature_config.type} feature {feature_config.name}")
        output_feature_class = get_from_registry(feature_config.type, get_output_type_registry())
        output_feature_obj = output_feature_class(feature_config, output_features=output_features)
        return output_feature_obj

    def get_model_inputs(self):
        """Returns a dict of feature name -> sample model input."""
        inputs = {
            input_feature_name: input_feature.create_sample_input()
            for input_feature_name, input_feature in self.input_features.items()
        }
        return inputs

    def get_model_size(self) -> int:
        """Returns total number of parameters in model."""
        model_tensors = self.collect_weights()
        total_size = 0
        for tnsr in model_tensors:
            total_size += tnsr[1].detach().cpu().numpy().size
        return total_size

    def to_torchscript(self, device: Optional[TorchDevice] = None):
        """Converts the ECD model as a TorchScript model."""
        if device is None:
            device = DEVICE

        self.eval()
        model_inputs = self.get_model_inputs()

        model_to_script = self.to(device)
        model_inputs_to_script = {k: v.to(device) for k, v in model_inputs.items()}
        # We set strict=False to enable dict inputs and outputs.
        return torch.jit.trace(model_to_script, model_inputs_to_script, strict=False)

    def save_torchscript(self, save_path, device: Optional[TorchDevice] = None):
        """Saves the ECD model as a TorchScript model."""
        if device is None:
            device = DEVICE

        traced = self.to_torchscript(device)
        traced.save(save_path)

    @property
    def input_shape(self):
        """Returns the shape of the model's input."""
        # TODO(justin): Remove dummy implementation. Make input_shape and output_shape functions.
        return torch.Size([1, 1])

    @abstractmethod
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

    def predictions(self, inputs):
        """Returns the model's predictions for the given inputs."""
        outputs = self(inputs)
        return self.outputs_to_predictions(outputs)

    def outputs_to_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns the model's predictions given the raw model outputs."""
        predictions = {}
        for of_name in self.output_features:
            predictions[of_name] = self.output_features[of_name].predictions(outputs, of_name)
        return predictions

    def evaluation_step(self, inputs, targets):
        """Predict the inputs and update evaluation metrics."""
        predictions = self.predictions(inputs)
        self.update_metrics(targets, predictions)
        return predictions

    def predict_step(self, inputs):
        """Predict the inputs."""
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
            train_loss += of_obj.loss.weight * of_train_loss
            of_train_losses[of_name] = of_train_loss

        additional_losses = self.losses()
        if additional_losses:
            train_loss += torch.sum(torch.stack(additional_losses))  # other losses

        # Add regularization loss
        if regularization_type is not None and regularization_lambda != 0:
            train_loss += reg_loss(self, regularization_type, l1=regularization_lambda, l2=regularization_lambda)

        return train_loss, of_train_losses

    def eval_loss(self, targets, predictions):
        """Computes all evaluation losses for the model given targets and predictions.

        Args:
            targets: A dictionary of target names to target tensors.
            predictions: A dictionary of output names to output tensors.

        Returns:
            A tuple of loss values for eval losses and additional losses.
        """
        eval_loss = 0
        for of_name, of_obj in self.output_features.items():
            of_eval_loss = of_obj.eval_loss(targets[of_name], predictions[of_name])
            eval_loss += of_obj.loss.weight * of_eval_loss

        additional_loss = 0
        additional_losses = self.losses()
        if additional_losses:
            additional_loss = torch.sum(torch.stack(additional_losses))  # other losses

        return eval_loss, additional_loss

    def update_metrics(self, targets, predictions):
        """Updates the model's metrics given targets and predictions."""
        for of_name, of_obj in self.output_features.items():
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        eval_loss, additional_losses = self.eval_loss(targets, predictions)
        self.eval_loss_metric.update(eval_loss)
        self.eval_additional_losses_metrics.update(additional_losses)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Returns a dictionary of metrics for each output feature of the model."""
        all_of_metrics = {}
        for of_name, of_obj in self.output_features.items():
            all_of_metrics[of_name] = of_obj.get_metrics()
        all_of_metrics[COMBINED] = {
            LOSS: get_scalar_from_ludwig_metric(self.eval_loss_metric)
            + get_scalar_from_ludwig_metric(self.eval_additional_losses_metrics)
        }
        return all_of_metrics

    def reset_metrics(self):
        """Resets the model's metrics."""
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

    def unskip(self):
        """Converts all skipped features into their fully encoded versions."""
        pass

    @abstractmethod
    def save(self, save_path: str):
        """Saves the model to the given path."""

    @abstractmethod
    def load(self, save_path: str):
        """Loads the model from the given path."""

    @abstractmethod
    def get_args(self):
        """Returns init arguments for constructing this model."""


def create_input_feature(feature_config: BaseInputFeatureConfig, encoder_obj: Optional[Encoder]) -> InputFeature:
    input_feature_cls = get_from_registry(feature_config.type, get_input_type_registry())
    input_feature = input_feature_cls(feature_config, encoder_obj=encoder_obj)
    if not feature_config.encoder.skip:
        return input_feature
    return create_passthrough_input_feature(input_feature, feature_config)
