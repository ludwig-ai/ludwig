import copy
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchmetrics

from ludwig.combiners.combiners import Combiner, get_combiner_class
from ludwig.constants import NAME, TIED, TYPE
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.abstractmodel import AbstractModel
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils import output_feature_utils
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


class ECD(AbstractModel):
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
            self.input_features.update(build_inputs(self._input_features_def))
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
        self.output_features.update(build_outputs(self._output_features_def, self.combiner))

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
