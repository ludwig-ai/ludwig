import copy
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchmetrics

from ludwig.combiners.combiners import Combiner, get_combiner_class
from ludwig.constants import COMBINED, LOSS, NAME, TIED, TYPE
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils import output_feature_utils
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.metric_utils import get_scalar_from_ludwig_metric
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule, reg_loss

logger = logging.getLogger(__name__)


class ECD(LudwigModule):
    def __init__(
        self,
        input_features_def,
        combiner_def,
        output_features_def,
        random_seed=None,
    ):
        self._input_features_def = copy.deepcopy(input_features_def)
        self._combiner_def = copy.deepcopy(combiner_def)
        self._output_features_def = copy.deepcopy(output_features_def)

        self._random_seed = random_seed

        # TODO: with change to misc_utils.set_random_seed() this may be redundant
        #       seems to be required for test_api.py::test_api_training_determinism
        if random_seed is not None:
            torch.random.manual_seed(random_seed)

        super().__init__()

        # ================ Inputs ================
        self.input_features = LudwigFeatureDict()
        try:
            self.input_features.update(build_inputs(self._input_features_def))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Combiner ================
        logger.debug(f"Combiner {combiner_def[TYPE]}")
        combiner_class = get_combiner_class(combiner_def[TYPE])
        config, kwargs = load_config_with_kwargs(
            combiner_class.get_schema_cls(),
            combiner_def,
        )
        self.combiner = combiner_class(input_features=self.input_features, config=config, **kwargs)

        # ================ Outputs ================
        self.output_features = LudwigFeatureDict()
        self.output_features.update(build_outputs(self._output_features_def, self.combiner))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        # After constructing all layers, clear the cache to free up memory
        clear_data_cache()

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
                inputs[input_feature_name] = input_values

        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs[input_feature_name] = encoder_output

        combiner_outputs = self.combiner(encoder_outputs)

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


def build_inputs(input_features_def: List[Dict[str, Any]]) -> Dict[str, InputFeature]:
    """Builds and returns input features in topological order."""
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def[NAME]] = build_single_input(input_feature_def, input_features)
    return input_features


def build_single_input(
    input_feature_def: Dict[str, Any], other_input_features: Dict[str, InputFeature]
) -> InputFeature:
    """Builds a single input feature from the input feature definition."""
    logger.debug(f"Input {input_feature_def[TYPE]} feature {input_feature_def[NAME]}")

    encoder_obj = None
    if input_feature_def.get(TIED, None) is not None:
        tied_input_feature_name = input_feature_def[TIED]
        if tied_input_feature_name in other_input_features:
            encoder_obj = other_input_features[tied_input_feature_name].encoder_obj

    input_feature_class = get_from_registry(input_feature_def[TYPE], input_type_registry)
    input_feature_obj = input_feature_class(input_feature_def, encoder_obj)

    return input_feature_obj


def build_outputs(output_features_def: List[Dict[str, Any]], combiner: Combiner) -> Dict[str, OutputFeature]:
    """Builds and returns output features in topological order."""
    output_features_def = topological_sort_feature_dependencies(output_features_def)
    output_features = {}

    for output_feature_def in output_features_def:
        # TODO(Justin): Check that the semantics of input_size align with what the combiner's output shape returns for
        # seq2seq.
        output_feature_def["input_size"] = combiner.output_shape[-1]
        output_feature = build_single_output(output_feature_def, output_features)
        output_features[output_feature_def[NAME]] = output_feature

    return output_features


def build_single_output(output_feature_def: Dict[str, Any], output_features: Dict[str, OutputFeature]) -> OutputFeature:
    """Builds a single output feature from the output feature definition."""
    logger.debug(f"Output {output_feature_def[TYPE]} feature {output_feature_def[NAME]}")

    output_feature_class = get_from_registry(output_feature_def[TYPE], output_type_registry)
    output_feature_obj = output_feature_class(output_feature_def, output_features)

    return output_feature_obj
