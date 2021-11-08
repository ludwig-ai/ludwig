import copy
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch
import torchmetrics

from ludwig.combiners.combiners import get_combiner_class
from ludwig.constants import *
from ludwig.features.base_feature import InputFeature
from ludwig.features.feature_registries import input_type_registry, \
    output_type_registry
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.schema_utils import load_config_with_kwargs
from ludwig.utils.torch_utils import LudwigModule, reg_loss
from ludwig.utils import output_feature_utils

logger = logging.getLogger(__name__)


class ECD(LudwigModule):

    def __init__(
            self,
            input_features_def,
            combiner_def,
            output_features_def,
            random_seed=None,
    ):
        # Deep copy to prevent TensorFlow from hijacking the dicts within the config and
        # transforming them into _DictWrapper classes, which are not JSON serializable.
        self._input_features_df = copy.deepcopy(input_features_def)
        self._combiner_def = copy.deepcopy(combiner_def)
        self._output_features_df = copy.deepcopy(output_features_def)

        self._random_seed = random_seed

        if random_seed is not None:
            torch.random.manual_seed(random_seed)

        super().__init__()

        # ================ Inputs ================
        self.input_features = torch.nn.ModuleDict()
        self.input_features.update(build_inputs(input_features_def))

        # ================ Combiner ================
        logger.debug('Combiner {}'.format(combiner_def[TYPE]))
        combiner_class = get_combiner_class(combiner_def[TYPE])
        config, kwargs = load_config_with_kwargs(
            combiner_class.get_schema_cls(),
            combiner_def,
        )
        self.combiner = combiner_class(
            input_features=self.input_features,
            config=config,
            **kwargs
        )

        # ================ Outputs ================
        self.output_features = torch.nn.ModuleDict()
        self.output_features.update(build_outputs(
            output_features_def, self.combiner))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()

        # After constructing all layers, clear the cache to free up memory
        clear_data_cache()

    def get_model_inputs(self):
        inputs = {
            input_feature_name: input_feature.create_sample_input()
            for input_feature_name, input_feature in
            self.input_features.items()
        }
        return inputs

    def save_torchscript(self, save_path):
        model_inputs = self.get_model_inputs()
        # We set strict=False to enable dict inputs and outputs.
        traced = torch.jit.trace(self, model_inputs, strict=False)
        traced.save(save_path)

    @property
    def input_shape(self):
        # TODO(justin): Remove dummy implementation. Make input_shape and output_shape functions.
        return torch.Size([1, 1])

    def forward(
            self,
            inputs: Union[
                Dict[str, torch.Tensor],
                Dict[str, np.ndarray],
                Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            ],
            mask=None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs: Inputs to the model. Can be a dictionary of input names to
                input tensors or a tuple of (inputs, targets) where inputs is
                a dictionary of input names to input tensors and targets is a
                dictionary of target names to target tensors.
            mask: A mask for the inputs.

        Returns:
            A dictionary of output {feature name}::{tensor_name} -> output tensor.
        """

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if isinstance(inputs, tuple):
            inputs, targets = inputs
            # Convert targets to tensors.
            for target_feature_name, target_value in targets.items():
                if not isinstance(target_value, torch.Tensor):
                    targets[target_feature_name] = torch.from_numpy(
                        target_value).to(device)
                else:
                    targets[target_feature_name] = target_value.to(device)
        else:
            targets = None

        assert inputs.keys() == self.input_features.keys()

        # Convert inputs to tensors.
        for input_feature_name, input_values in inputs.items():
            if not isinstance(input_values, torch.Tensor):
                inputs[input_feature_name] = torch.from_numpy(input_values).to(device)
            else:
                inputs[input_feature_name] = input_values.to(device)

        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs[input_feature_name] = encoder_output

        combiner_outputs = self.combiner(encoder_outputs)

        output_logits = {}
        output_last_hidden = {}
        for output_feature_name, decoder in self.output_features.items():
            # use presence or absence of targets
            # to signal training or prediction
            decoder_inputs = (combiner_outputs, copy.copy(output_last_hidden))
            if targets is not None:
                # targets are only used during training,
                # during prediction they are omitted
                decoder_inputs = (decoder_inputs, targets[output_feature_name])

            decoder_outputs = decoder(decoder_inputs, mask=mask)

            # Add decoder outputs to overall output dictionary.
            for decoder_output_name, tensor in decoder_outputs.items():
                output_feature_utils.set_output_feature_tensor(
                    output_logits, output_feature_name, decoder_output_name, tensor)
        return output_logits

    def predictions(self, inputs, output_features=None):
        # check validity of output_features
        if output_features is None:
            of_list = self.output_features
        elif isinstance(output_features, str):
            if output_features == 'all':
                of_list = set(self.output_features.keys())
            elif output_features in self.output_features:
                of_list = [output_features]
            else:
                raise ValueError(
                    "'output_features' {} is not a valid for this model. "
                    "Available ones are: {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        elif isinstance(output_features, list or set):
            if output_features.issubset(self.output_features):
                of_list = output_features
            else:
                raise ValueError(
                    "'output_features' {} must be a subset of "
                    "available features {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        else:
            raise ValueError(
                "'output_features' must be None or a string or a list "
                "of output features"
            )

        outputs = self(inputs)

        predictions = {}
        for of_name in of_list:
            predictions[of_name] = self.output_features[of_name].predictions(
                outputs, of_name)

        return predictions

    def evaluation_step(self, inputs, targets):
        predictions = self.predictions(inputs, output_features=None)
        self.update_metrics(targets, predictions)
        return predictions

    def predict_step(self, inputs):
        return self.predictions(inputs, output_features=None)

    def train_loss(
            self,
            targets,
            predictions,
            regularization_type: Optional[str] = None,
            regularization_lambda: Optional[float] = None
    ):
        train_loss = 0
        of_train_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_train_loss = of_obj.train_loss(
                targets[of_name], predictions, of_name)
            train_loss += of_obj.loss['weight'] * of_train_loss
            of_train_losses[of_name] = of_train_loss

        for loss in self.losses():
            train_loss += loss

        # Add regularization loss
        if regularization_type is not None:
            train_loss += reg_loss(
                self,
                regularization_type,
                l1=regularization_lambda,
                l2=regularization_lambda
            )

        return train_loss, of_train_losses

    def eval_loss(self, targets, predictions):
        eval_loss = 0
        of_eval_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_eval_loss = of_obj.eval_loss(
                targets[of_name], predictions[of_name]
            )
            eval_loss += of_obj.loss['weight'] * of_eval_loss
            of_eval_losses[of_name] = of_eval_loss
        eval_loss += sum(self.losses())  # regularization / other losses
        return eval_loss, of_eval_losses

    def update_metrics(self, targets, predictions):
        for of_name, of_obj in self.output_features.items():
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        self.eval_loss_metric.update(self.eval_loss(targets, predictions)[0])

    def get_metrics(self):
        all_of_metrics = {}
        for of_name, of_obj in self.output_features.items():
            all_of_metrics[of_name] = of_obj.get_metrics()
        all_of_metrics[COMBINED] = {
            LOSS: self.eval_loss_metric.compute().detach().numpy().item()
        }
        return all_of_metrics

    def reset_metrics(self):
        for of_obj in self.output_features.values():
            of_obj.reset_metrics()
        self.eval_loss_metric.reset()

    def collect_weights(
            self,
            tensor_names=None,
            **kwargs
    ):
        """Returns named parameters filtered against `tensor_names` if not None."""
        if not tensor_names:
            return self.named_parameters()

        # Check for bad tensor names.
        weight_names = set(name for name, _ in self.named_parameters())
        for name in tensor_names:
            if name not in weight_names:
                raise ValueError(
                    f'Requested tensor name filter "{name}" not present in the model graph')

        # Apply filter.
        tensor_set = set(tensor_names)
        return [named_param for named_param in self.named_parameters() if named_param[0] in tensor_set]

    def get_args(self):
        return self._input_features_df, self._combiner_def, self._output_features_df, self._random_seed


def build_inputs(
        input_features_def: List[Dict[str, Any]],
        **kwargs
) -> Dict[str, InputFeature]:
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(
        input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def[NAME]] = build_single_input(
            input_feature_def,
            input_features,
            **kwargs
        )
    return input_features


def build_single_input(
        input_feature_def: Dict[str, Any],
        other_input_features: Dict[str, InputFeature],
        **kwargs
) -> InputFeature:
    logger.debug('Input {} feature {}'.format(
        input_feature_def[TYPE],
        input_feature_def[NAME]
    ))

    encoder_obj = None
    if input_feature_def.get(TIED, None) is not None:
        tied_input_feature_name = input_feature_def[TIED]
        if tied_input_feature_name in other_input_features:
            encoder_obj = other_input_features[
                tied_input_feature_name].encoder_obj

    input_feature_class = get_from_registry(
        input_feature_def[TYPE],
        input_type_registry
    )
    input_feature_obj = input_feature_class(input_feature_def, encoder_obj)

    return input_feature_obj


def build_outputs(
        output_features_def,
        combiner,
        **kwargs
):
    output_features_def = topological_sort_feature_dependencies(
        output_features_def)
    output_features = {}

    for output_feature_def in output_features_def:
        output_feature_def["input_size"] = combiner.output_shape[-1]
        output_feature = build_single_output(
            output_feature_def,
            combiner,
            output_features,
            **kwargs
        )
        output_features[output_feature_def[NAME]] = output_feature

    return output_features


def build_single_output(
        output_feature_def,
        feature_hidden,
        other_output_features,
        **kwargs
):
    logger.debug('Output {} feature {}'.format(
        output_feature_def[TYPE],
        output_feature_def[NAME]
    ))

    output_feature_class = get_from_registry(
        output_feature_def[TYPE],
        output_type_registry
    )
    output_feature_obj = output_feature_class(output_feature_def)
    # weighted_train_mean_loss, weighted_eval_loss, output_tensors = output_feature_obj.concat_dependencies_and_build_output(
    #    feature_hidden,
    #    other_output_features,
    #    **kwargs
    # )

    return output_feature_obj
