#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Type

import torch
from torch.nn import Linear, ModuleList

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, ENCODER_OUTPUT, NUMBER
from ludwig.encoders.registry import get_sequence_encoder_registry
from ludwig.features.base_feature import InputFeature
from ludwig.modules.attention_modules import TransformerStack
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.modules.tabnet_modules import TabNet
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.comparator import ComparatorCombinerConfig
from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.project_aggregate import ProjectAggregateCombinerConfig
from ludwig.schema.combiners.sequence import SequenceCombinerConfig
from ludwig.schema.combiners.sequence_concat import SequenceConcatCombinerConfig
from ludwig.schema.combiners.tab_transformer import TabTransformerCombinerConfig
from ludwig.schema.combiners.tabnet import TabNetCombinerConfig
from ludwig.schema.combiners.transformer import TransformerCombinerConfig
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.registry import Registry
from ludwig.utils.torch_utils import LudwigModule, sequence_length_3D
from ludwig.utils.torch_utils import sequence_mask as torch_sequence_mask

logger = logging.getLogger(__name__)


@dataclass
class Handle:
    """This class provides an opaque handle to the input features, preventing them from being registered as state.

    This is important because we already reference the `input_features` as an attribute of ECD, so we don't need it to
    appear twice in the state_dict. Furthermore, DeepSpeed will get terribly confused if have the input features set as
    an attribute of the combiner, and lead to shape mismatch errors when we go to load a saved checkpoint.
    """

    input_features: Dict[str, "InputFeature"]


@DeveloperAPI
class Combiner(LudwigModule, ABC):
    """Base class for combiners, which implements common properties.

    Subclasses will usually override:     __init__()        to set properties and allocate resources.  Should call
    super().__init__(input_features).     forward()         performs the forward pass given a dictionary of encoder
    outputs.     get_schema_cls()  must returns the class of the corresponding schema for the combiner type.
    """

    def __init__(self, input_features: Dict[str, "InputFeature"]):
        super().__init__()
        self.handle = Handle(input_features)

    @property
    def concatenated_shape(self) -> torch.Size:
        # compute the size of the last dimension for the incoming encoder outputs
        # this is required to setup the fully connected layer
        shapes = [
            torch.prod(torch.Tensor([*self.handle.input_features.get(k).output_shape]))
            for k in self.handle.input_features
        ]
        return torch.Size([torch.sum(torch.Tensor(shapes)).type(torch.int32)])

    @property
    def input_shape(self) -> Dict:
        # input to combiner is a dictionary of the input features encoder
        # outputs, this property returns dictionary of output shapes for each
        # input feature's encoder output shapes.
        return {k: self.handle.input_features.get(k).output_shape for k in self.handle.input_features}

    @property
    @lru_cache(maxsize=1)
    def output_shape(self) -> torch.Size:
        pseudo_input = {}
        for k in self.handle.input_features:
            pseudo_input[k] = {
                ENCODER_OUTPUT: torch.rand(
                    2, *self.handle.input_features.get(k).output_shape, dtype=self.input_dtype, device=self.device
                )
            }
        output_tensor = self.forward(pseudo_input)
        return output_tensor["combiner_output"].size()[1:]


combiner_impl_registry = Registry[Type[Combiner]]()


def register_combiner(config_cls: Type[BaseCombinerConfig]):
    def wrap(cls: Type[Combiner]):
        combiner_impl_registry[config_cls] = cls
        return cls

    return wrap


def create_combiner(config: BaseCombinerConfig, **kwargs) -> Combiner:
    return combiner_impl_registry[type(config)](config=config, **kwargs)


@register_combiner(ConcatCombinerConfig)
class ConcatCombiner(Combiner):
    def __init__(self, input_features: Dict[str, "InputFeature"] = None, config: ConcatCombinerConfig = None, **kwargs):
        super().__init__(input_features)
        self.name = "ConcatCombiner"
        logger.debug(f" {self.name}")

        self.flatten_inputs = config.flatten_inputs
        self.fc_stack = None

        # todo future: this may be redundant, check
        fc_layers = config.fc_layers
        if fc_layers is None:
            fc_layers = []
            for i in range(config.num_fc_layers):
                fc_layers.append({"output_size": config.output_size})
        self.fc_layers = fc_layers

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=self.concatenated_shape[-1],
            layers=config.fc_layers,
            num_layers=config.num_fc_layers,
            default_output_size=config.output_size,
            default_use_bias=config.use_bias,
            default_weights_initializer=config.weights_initializer,
            default_bias_initializer=config.bias_initializer,
            default_norm=config.norm,
            default_norm_params=config.norm_params,
            default_activation=config.activation,
            default_dropout=config.dropout,
            residual=config.residual,
        )

        if input_features and len(input_features) == 1 and self.fc_layers is None:
            self.supports_masking = True

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs]

        # ================ Flatten ================
        if self.flatten_inputs:
            batch_size = encoder_outputs[0].shape[0]
            encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs]

        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = torch.cat(encoder_outputs, 1)
        else:
            hidden = list(encoder_outputs)[0]

        # ================ Fully Connected ================
        hidden = self.fc_stack(hidden)

        return_data = {"combiner_output": hidden}

        if len(inputs) == 1:
            # Workaround for including additional tensors from output of input encoders for
            # potential use in decoders, e.g. LSTM state for seq2seq.
            # TODO(Justin): Think about how to make this communication work for multi-sequence
            # features. Other combiners.
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data


@register_combiner(SequenceConcatCombinerConfig)
class SequenceConcatCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"], config: SequenceConcatCombinerConfig = None, **kwargs
    ):
        super().__init__(input_features)
        self.name = "SequenceConcatCombiner"
        logger.debug(f" {self.name}")

        self.reduce_output = config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=config.reduce_output,
            max_sequence_length=self.concatenated_shape[0],
            encoding_size=self.concatenated_shape[1],
        )
        if self.reduce_output is None:
            self.supports_masking = True
        self.main_sequence_feature = config.main_sequence_feature

    @property
    def concatenated_shape(self) -> torch.Size:
        # computes the effective shape of the input tensor after combining
        # all the encoder outputs
        # determine max sequence length by finding the first sequence tensor
        # assume all the sequences are of the same size, if not true
        # this will be caught during processing
        seq_size = None
        for k in self.handle.input_features:
            # dim-2 output_shape implies a sequence [seq_size, hidden]
            if len(self.handle.input_features.get(k).output_shape) == 2:
                seq_size = self.handle.input_features.get(k).output_shape[0]
                break

        # collect the size of the last dimension for all input feature
        # encoder outputs
        shapes = [
            self.handle.input_features.get(k).output_shape[-1] for k in self.handle.input_features
        ]  # output shape not input shape
        return torch.Size([seq_size, sum(shapes)])

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        if self.main_sequence_feature is None or self.main_sequence_feature not in inputs:
            for if_name, if_outputs in inputs.items():
                # todo: when https://github.com/ludwig-ai/ludwig/issues/810 is closed
                #       convert following test from using shape to use explicit
                #       if_outputs[TYPE] values for sequence features
                if len(if_outputs[ENCODER_OUTPUT].shape) == 3:
                    self.main_sequence_feature = if_name
                    break

        if self.main_sequence_feature is None:
            raise Exception("No sequence feature available for sequence combiner")

        main_sequence_feature_encoding = inputs[self.main_sequence_feature]

        representation = main_sequence_feature_encoding[ENCODER_OUTPUT]
        representations = [representation]

        sequence_max_length = representation.shape[1]
        sequence_length = sequence_length_3D(representation)

        # ================ Concat ================
        for if_name, if_outputs in inputs.items():
            if if_name != self.main_sequence_feature:
                if_representation = if_outputs[ENCODER_OUTPUT]
                if len(if_representation.shape) == 3:
                    # The following check makes sense when
                    # both representations have a specified
                    # sequence length dimension. If they do not,
                    # then this check is simply checking if None == None
                    # and will not catch discrepancies in the different
                    # feature length dimension. Those errors will show up
                    # at training time. Possible solutions to this is
                    # to enforce a length second dimension in
                    # sequential feature placeholders, but that
                    # does not work with BucketedBatcher that requires
                    # the second dimension to be undefined in order to be
                    # able to trim the data points and speed up computation.
                    # So for now we are keeping things like this, make sure
                    # to write in the documentation that training time
                    # dimensions mismatch may occur if the sequential
                    # features have different lengths for some data points.
                    if if_representation.shape[1] != representation.shape[1]:
                        raise ValueError(
                            "The sequence length of the input feature {} "
                            "is {} and is different from the sequence "
                            "length of the main sequence feature {} which "
                            "is {}.\n Shape of {}: {}, shape of {}: {}.\n"
                            "Sequence lengths of all sequential features "
                            "must be the same  in order to be concatenated "
                            "by the sequence concat combiner. "
                            "Try to impose the same max sequence length "
                            "as a preprocessing parameter to both features "
                            "or to reduce the output of {}.".format(
                                if_name,
                                if_representation.shape[1],
                                self.main_sequence_feature,
                                representation.shape[1],
                                if_name,
                                if_representation.shape,
                                if_name,
                                representation.shape,
                                if_name,
                            )
                        )
                    # this assumes all sequence representations have the
                    # same sequence length, 2nd dimension
                    representations.append(if_representation)

                elif len(if_representation.shape) == 2:
                    multipliers = (1, sequence_max_length, 1)
                    tiled_representation = torch.tile(torch.unsqueeze(if_representation, 1), multipliers)
                    representations.append(tiled_representation)

                else:
                    raise ValueError(
                        "The representation of {} has rank {} and cannot be"
                        " concatenated by a sequence concat combiner. "
                        "Only rank 2 and rank 3 tensors are supported.".format(if_name, len(if_representation.shape))
                    )

        hidden = torch.cat(representations, 2)
        logger.debug(f"  concat_hidden: {hidden}")

        # ================ Mask ================
        sequence_mask = torch_sequence_mask(sequence_length, sequence_max_length)
        hidden = torch.multiply(hidden, torch.unsqueeze(sequence_mask, -1).type(torch.float32))

        # ================ Reduce ================
        hidden = self.reduce_sequence(hidden)

        return_data = {"combiner_output": hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data


@register_combiner(SequenceCombinerConfig)
class SequenceCombiner(Combiner):
    def __init__(self, input_features: Dict[str, "InputFeature"], config: SequenceCombinerConfig = None, **kwargs):
        super().__init__(input_features)
        self.name = "SequenceCombiner"
        logger.debug(f" {self.name}")

        self.combiner = SequenceConcatCombiner(
            input_features,
            config=SequenceConcatCombinerConfig(reduce_output=None, main_sequence_feature=config.main_sequence_feature),
        )

        logger.debug(
            f"combiner input shape {self.combiner.concatenated_shape}, " f"output shape {self.combiner.output_shape}"
        )

        self.encoder_obj = get_from_registry(config.encoder.type, get_sequence_encoder_registry())(
            should_embed=False,
            reduce_output=config.reduce_output,
            embedding_size=self.combiner.output_shape[1],
            max_sequence_length=self.combiner.output_shape[0],
            **kwargs,
        )

        if hasattr(self.encoder_obj, "supports_masking") and self.encoder_obj.supports_masking:
            self.supports_masking = True

    @property
    def concatenated_shape(self) -> torch.Size:
        # computes the effective shape of the input tensor after combining
        # all the encoder outputs
        # determine max sequence length by finding the first sequence tensor
        # assume all the sequences are of the same size, if not true
        # this will be caught during processing
        seq_size = None
        for k in self.handle.input_features:
            # dim-2 output_shape implies a sequence [seq_size, hidden]
            if len(self.handle.input_features.get(k).output_shape) == 2:
                seq_size = self.handle.input_features.get(k).output_shape[0]
                break

        # collect the size of the last dimension for all input feature
        # encoder outputs
        shapes = [
            self.handle.input_features.get(k).output_shape[-1] for k in self.handle.input_features
        ]  # output shape not input shape
        return torch.Size([seq_size, sum(shapes)])

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        # ================ Concat ================
        hidden = self.combiner(inputs)

        # ================ Sequence encoding ================
        hidden = self.encoder_obj(hidden["combiner_output"])

        return_data = {"combiner_output": hidden[ENCODER_OUTPUT]}
        for key, value in hidden.items():
            if key != ENCODER_OUTPUT:
                return_data[key] = value

        return return_data


@register_combiner(TabNetCombinerConfig)
class TabNetCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"], config: TabNetCombinerConfig = None, **kwargs
    ) -> None:
        super().__init__(input_features)
        self.name = "TabNetCombiner"
        logger.debug(f" {self.name}")

        self.tabnet = TabNet(
            self.concatenated_shape[-1],
            config.size,
            config.output_size,
            num_steps=config.num_steps,
            num_total_blocks=config.num_total_blocks,
            num_shared_blocks=config.num_shared_blocks,
            relaxation_factor=config.relaxation_factor,
            bn_epsilon=config.bn_epsilon,
            bn_momentum=config.bn_momentum,
            bn_virtual_bs=config.bn_virtual_bs,
            sparsity=config.sparsity,
            entmax_mode=config.entmax_mode,
            entmax_alpha=config.entmax_alpha,
        )

        if config.dropout > 0:
            self.dropout = torch.nn.Dropout(config.dropout)
        else:
            self.dropout = None

    @property
    def concatenated_shape(self) -> torch.Size:
        # compute the size of the last dimension for the incoming encoder outputs
        # this is required to setup
        shapes = [
            torch.prod(torch.Tensor([*self.handle.input_features.get(k).output_shape]))
            for k in self.handle.input_features
        ]
        return torch.Size([torch.sum(torch.Tensor(shapes)).type(torch.int32)])

    def forward(
        self,
        inputs: torch.Tensor,  # encoder outputs
    ) -> Dict:
        encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs]

        # ================ Flatten ================
        batch_size = encoder_outputs[0].shape[0]
        encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs]

        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = torch.cat(encoder_outputs, 1)
        else:
            hidden = list(encoder_outputs)[0]

        # ================ TabNet ================
        hidden, aggregated_mask, masks = self.tabnet(hidden)
        if self.dropout:
            hidden = self.dropout(hidden)

        return_data = {
            "combiner_output": hidden,
            "aggregated_attention_masks": aggregated_mask,
            "attention_masks": masks,
        }

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data

    @property
    def output_shape(self) -> torch.Size:
        return self.tabnet.output_shape


@register_combiner(TransformerCombinerConfig)
class TransformerCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"] = None, config: TransformerCombinerConfig = None, **kwargs
    ):
        super().__init__(input_features)
        self.name = "TransformerCombiner"
        logger.debug(f" {self.name}")

        self.reduce_output = config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=config.reduce_output,
            max_sequence_length=len(input_features),
            encoding_size=config.hidden_size,
        )
        if self.reduce_output is None:
            self.supports_masking = True

        # max sequence length for Transformer layer is number of input features
        self.max_sequence_length = len(input_features)

        logger.debug("  Projectors")
        self.projectors = ModuleList(
            # regardless of rank-2 or rank-3 input, torch.prod() calculates size
            # after flattening the encoder output tensor
            [
                Linear(
                    torch.prod(torch.Tensor([*input_features.get(inp).output_shape])).type(torch.int32),
                    config.hidden_size,
                )
                for inp in input_features
            ]
        )

        logger.debug("  TransformerStack")
        self.transformer_stack = TransformerStack(
            input_size=config.hidden_size,
            max_sequence_length=self.max_sequence_length,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            output_size=config.transformer_output_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.transformer_stack.output_shape[-1],
                layers=config.fc_layers,
                num_layers=config.num_fc_layers,
                default_output_size=config.output_size,
                default_use_bias=config.use_bias,
                default_weights_initializer=config.weights_initializer,
                default_bias_initializer=config.bias_initializer,
                default_norm=config.norm,
                default_norm_params=config.norm_params,
                default_activation=config.fc_activation,
                default_dropout=config.fc_dropout,
                fc_residual=config.fc_residual,
            )

    def forward(
        self,
        inputs,  # encoder outputs
    ) -> Dict:
        encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs]

        # ================ Flatten ================
        batch_size = encoder_outputs[0].shape[0]
        encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs]

        # ================ Project & Concat ================
        projected = [self.projectors[i](eo) for i, eo in enumerate(encoder_outputs)]
        hidden = torch.stack(projected)  # shape [num_eo, bs, h]
        hidden = torch.permute(hidden, (1, 0, 2))  # shape [bs, num_eo, h]

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(hidden)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden)

        return_data = {"combiner_output": hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data


@register_combiner(TabTransformerCombinerConfig)
class TabTransformerCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"] = None, config: TabTransformerCombinerConfig = None, **kwargs
    ):
        super().__init__(input_features)
        self.name = "TabTransformerCombiner"
        logger.debug(f"Initializing {self.name}")

        self.reduce_output = config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=config.reduce_output, max_sequence_length=len(input_features), encoding_size=config.hidden_size
        )
        self.supports_masking = True

        self.embed_input_feature_name = config.embed_input_feature_name
        if self.embed_input_feature_name:
            vocab = [
                i_f
                for i_f in input_features
                if input_features.get(i_f).type() != NUMBER or input_features.get(i_f).type() != BINARY
            ]
            if self.embed_input_feature_name == "add":
                self.embed_i_f_name_layer = Embed(vocab, config.hidden_size, force_embedding_size=True)
                projector_size = config.hidden_size
            elif isinstance(self.embed_input_feature_name, int):
                if self.embed_input_feature_name > config.hidden_size:
                    raise ValueError(
                        "TabTransformer parameter "
                        "`embed_input_feature_name` "
                        "specified integer value ({}) "
                        "needs to be smaller than "
                        "`hidden_size` ({}).".format(self.embed_input_feature_name, config.hidden_size)
                    )
                self.embed_i_f_name_layer = Embed(
                    vocab,
                    self.embed_input_feature_name,
                    force_embedding_size=True,
                )
                projector_size = config.hidden_size - self.embed_input_feature_name
            else:
                raise ValueError(
                    "TabTransformer parameter "
                    "`embed_input_feature_name` "
                    "should be either None, an integer or `add`, "
                    "the current value is "
                    "{}".format(self.embed_input_feature_name)
                )
        else:
            projector_size = config.hidden_size

        logger.debug("  Projectors")
        self.unembeddable_features = []
        self.embeddable_features = []
        for i_f in input_features:
            if input_features.get(i_f).type() in {NUMBER, BINARY}:
                self.unembeddable_features.append(i_f)
            else:
                self.embeddable_features.append(i_f)

        self.projectors = ModuleList()
        for i_f in self.embeddable_features:
            flatten_size = self.get_flatten_size(input_features.get(i_f).output_shape)
            self.projectors.append(Linear(flatten_size[0], projector_size))

        # input to layer_norm are the encoder outputs for unembeddable features,
        # which are number or binary features.  These should be 2-dim
        # tensors.  Size should be concatenation of these tensors.
        concatenated_unembeddable_encoders_size = 0
        for i_f in self.unembeddable_features:
            concatenated_unembeddable_encoders_size += input_features.get(i_f).output_shape[0]

        self.layer_norm = torch.nn.LayerNorm(concatenated_unembeddable_encoders_size)

        logger.debug("  TransformerStack")
        self.transformer_stack = TransformerStack(
            input_size=config.hidden_size,
            max_sequence_length=len(self.embeddable_features),
            hidden_size=config.hidden_size,
            # todo: can we just use projector_size? # hidden_size,
            num_heads=config.num_heads,
            output_size=config.transformer_output_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        logger.debug("  FCStack")

        # determine input size to fully connected layer based on reducer
        if config.reduce_output == "concat":
            fc_input_size = len(self.embeddable_features) * config.hidden_size
        else:
            fc_input_size = self.reduce_sequence.output_shape[-1] if len(self.embeddable_features) > 0 else 0
        self.fc_stack = FCStack(
            fc_input_size + concatenated_unembeddable_encoders_size,
            layers=config.fc_layers,
            num_layers=config.num_fc_layers,
            default_output_size=config.output_size,
            default_use_bias=config.use_bias,
            default_weights_initializer=config.weights_initializer,
            default_bias_initializer=config.bias_initializer,
            default_norm=config.norm,
            default_norm_params=config.norm_params,
            default_activation=config.fc_activation,
            default_dropout=config.fc_dropout,
            fc_residual=config.fc_residual,
        )

        self._empty_hidden = torch.empty([1, 0])
        self._embeddable_features_indices = torch.arange(0, len(self.embeddable_features))

        # Create empty tensor of shape [1, 0] to use as hidden in case there are no category or numeric/binary features.
        self.register_buffer("empty_hidden", self._empty_hidden)
        self.register_buffer("embeddable_features_indices", self._embeddable_features_indices)

    @staticmethod
    def get_flatten_size(output_shape: torch.Size) -> torch.Size:
        size = torch.prod(torch.Tensor([*output_shape]))
        return torch.Size([size.type(torch.int32)])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    def forward(
        self,
        inputs: Dict,  # encoder outputs
    ) -> Dict:
        unembeddable_encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs if k in self.unembeddable_features]
        embeddable_encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs if k in self.embeddable_features]

        batch_size = (
            embeddable_encoder_outputs[0].shape[0]
            if len(embeddable_encoder_outputs) > 0
            else unembeddable_encoder_outputs[0].shape[0]
        )

        # ================ Project & Concat embeddables ================
        if len(embeddable_encoder_outputs) > 0:
            # ============== Flatten =================
            embeddable_encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in embeddable_encoder_outputs]

            projected = [self.projectors[i](eo) for i, eo in enumerate(embeddable_encoder_outputs)]
            hidden = torch.stack(projected)  # num_eo, bs, h
            hidden = torch.permute(hidden, (1, 0, 2))  # bs, num_eo, h

            if self.embed_input_feature_name:
                i_f_names_idcs = torch.reshape(
                    torch.arange(0, len(embeddable_encoder_outputs), device=self.device), [-1, 1]
                )
                embedded_i_f_names = self.embed_i_f_name_layer(i_f_names_idcs)
                embedded_i_f_names = torch.unsqueeze(embedded_i_f_names, dim=0)
                embedded_i_f_names = torch.tile(embedded_i_f_names, [batch_size, 1, 1])
                if self.embed_input_feature_name == "add":
                    hidden = hidden + embedded_i_f_names
                else:
                    hidden = torch.cat([hidden, embedded_i_f_names], -1)

            # ================ Transformer Layers ================
            hidden = self.transformer_stack(hidden)

            # ================ Sequence Reduction ================
            hidden = self.reduce_sequence(hidden)
        else:
            # create empty tensor because there are no category features
            hidden = torch.empty([batch_size, 0], device=self.device)

        # ================ Concat Skipped ================
        if len(unembeddable_encoder_outputs) > 0:
            unembeddable_encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in unembeddable_encoder_outputs]
            # ================ Flatten ================
            if len(unembeddable_encoder_outputs) > 1:
                unembeddable_hidden = torch.cat(unembeddable_encoder_outputs, -1)  # tf.keras.layers.concatenate
            else:
                unembeddable_hidden = list(unembeddable_encoder_outputs)[0]
            unembeddable_hidden = self.layer_norm(unembeddable_hidden)

        else:
            # create empty tensor because there are not numeric/binary features
            unembeddable_hidden = torch.tile(self.empty_hidden, [batch_size, 0])

        # ================ Concat Skipped and Others ================
        hidden = torch.cat([hidden, unembeddable_hidden], -1)

        # ================ FC Layers ================
        hidden = self.fc_stack(hidden)

        return_data = {"combiner_output": hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data


@register_combiner(ComparatorCombinerConfig)
class ComparatorCombiner(Combiner):
    def __init__(
        self,
        input_features: Dict[str, "InputFeature"],
        config: ComparatorCombinerConfig = None,
        **kwargs,
    ):
        super().__init__(input_features)
        self.name = "ComparatorCombiner"
        logger.debug(f"Entering {self.name}")

        self.entity_1 = config.entity_1
        self.entity_2 = config.entity_2
        self.required_inputs = set(config.entity_1 + config.entity_2)
        self.output_size = config.output_size

        self.fc_stack = None

        # todo future: this may be redundant, check
        fc_layers = config.fc_layers
        if fc_layers is None and config.num_fc_layers is not None:
            fc_layers = []
            for _ in range(config.num_fc_layers):
                fc_layers.append({"output_size": config.output_size})

        if fc_layers is not None:
            logger.debug("Setting up FCStack")
            self.e1_fc_stack = FCStack(
                self.get_entity_shape(config.entity_1)[-1],
                layers=fc_layers,
                num_layers=config.num_fc_layers,
                default_output_size=config.output_size,
                default_use_bias=config.use_bias,
                default_weights_initializer=config.weights_initializer,
                default_bias_initializer=config.bias_initializer,
                default_norm=config.norm,
                default_norm_params=config.norm_params,
                default_activation=config.activation,
                default_dropout=config.dropout,
            )
            self.e2_fc_stack = FCStack(
                self.get_entity_shape(config.entity_2)[-1],
                layers=fc_layers,
                num_layers=config.num_fc_layers,
                default_output_size=config.output_size,
                default_use_bias=config.use_bias,
                default_weights_initializer=config.weights_initializer,
                default_bias_initializer=config.bias_initializer,
                default_norm=config.norm,
                default_norm_params=config.norm_params,
                default_activation=config.activation,
                default_dropout=config.dropout,
            )

        self.last_fc_layer_output_size = fc_layers[-1]["output_size"]

        # todo: set initializer and regularization
        self.register_buffer(
            "bilinear_weights",
            torch.randn([self.last_fc_layer_output_size, self.last_fc_layer_output_size], dtype=torch.float32),
        )

    def get_entity_shape(self, entity: list) -> torch.Size:
        sizes = [torch.prod(torch.Tensor([*self.handle.input_features.get(k).output_shape])) for k in entity]
        return torch.Size([torch.sum(torch.Tensor(sizes)).type(torch.int32)])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([2 * self.last_fc_layer_output_size + 2])

    def forward(
        self,
        inputs: Dict,  # encoder outputs
    ) -> Dict[str, torch.Tensor]:  # encoder outputs
        if inputs.keys() != self.required_inputs:
            raise ValueError(f"Missing inputs {self.required_inputs - set(inputs.keys())}")

        ############
        # Entity 1 #
        ############
        e1_enc_outputs = [inputs[k][ENCODER_OUTPUT] for k in self.entity_1]

        # ================ Flatten ================
        batch_size = e1_enc_outputs[0].shape[0]
        e1_enc_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in e1_enc_outputs]

        # ================ Concat ================
        if len(e1_enc_outputs) > 1:
            e1_hidden = torch.cat(e1_enc_outputs, 1)
        else:
            e1_hidden = list(e1_enc_outputs)[0]

        # ================ Fully Connected ================
        e1_hidden = self.e1_fc_stack(e1_hidden)  # [bs, output_size]

        ############
        # Entity 2 #
        ############
        e2_enc_outputs = [inputs[k][ENCODER_OUTPUT] for k in self.entity_2]

        # ================ Flatten ================
        batch_size = e2_enc_outputs[0].shape[0]
        e2_enc_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in e2_enc_outputs]

        # ================ Concat ================
        if len(e2_enc_outputs) > 1:
            e2_hidden = torch.cat(e2_enc_outputs, 1)
        else:
            e2_hidden = list(e2_enc_outputs)[0]

        # ================ Fully Connected ================
        e2_hidden = self.e2_fc_stack(e2_hidden)  # [bs, output_size]

        ###########
        # Compare #
        ###########
        if e1_hidden.shape != e2_hidden.shape:
            raise ValueError(
                f"Mismatching shapes among dimensions! "
                f"entity1 shape: {e1_hidden.shape} "
                f"entity2 shape: {e2_hidden.shape}"
            )

        element_wise_mul = e1_hidden * e2_hidden  # [bs, output_size]
        dot_product = torch.sum(element_wise_mul, 1, keepdim=True)  # [bs, 1]
        abs_diff = torch.abs(e1_hidden - e2_hidden)  # [bs, output_size]
        bilinear_prod = torch.bmm(
            torch.mm(e1_hidden, self.bilinear_weights).unsqueeze(1), e2_hidden.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [bs, 1]

        logger.debug(
            "preparing combiner output by concatenating these tensors: "
            f"dot_product: {dot_product.shape}, element_size_mul: {element_wise_mul.shape}"
            f", abs_diff: {abs_diff.shape}, bilinear_prod {bilinear_prod.shape}"
        )
        hidden = torch.cat([dot_product, element_wise_mul, abs_diff, bilinear_prod], 1)  # [bs, 2 * output_size + 2]

        return {"combiner_output": hidden}


@register_combiner(ProjectAggregateCombinerConfig)
class ProjectAggregateCombiner(Combiner):
    def __init__(
        self, input_features: Dict[str, "InputFeature"] = None, config: ProjectAggregateCombinerConfig = None, **kwargs
    ):
        super().__init__(input_features)
        self.name = "ProjectAggregateCombiner"
        logger.debug(f" {self.name}")

        logger.debug("  Projectors")
        self.projectors = ModuleList(
            # regardless of rank-2 or rank-3 input, torch.prod() calculates size
            # after flattening the encoder output tensor
            [
                Linear(
                    torch.prod(torch.Tensor([*input_features.get(inp).output_shape])).type(torch.int32),
                    config.projection_size,
                )
                for inp in input_features
            ]
        )

        self.fc_stack = None

        # todo future: this may be redundant, check
        fc_layers = config.fc_layers
        if fc_layers is None and config.num_fc_layers is not None:
            fc_layers = []
            for i in range(config.num_fc_layers):
                fc_layers.append({"output_size": config.output_size})

        self.fc_layers = fc_layers
        if self.fc_layers is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                first_layer_input_size=config.projection_size,
                layers=config.fc_layers,
                num_layers=config.num_fc_layers,
                default_output_size=config.output_size,
                default_use_bias=config.use_bias,
                default_weights_initializer=config.weights_initializer,
                default_bias_initializer=config.bias_initializer,
                default_norm=config.norm,
                default_norm_params=config.norm_params,
                default_activation=config.activation,
                default_dropout=config.dropout,
                residual=config.residual,
            )

        if input_features and len(input_features) == 1 and self.fc_layers is None:
            self.supports_masking = True

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs]

        # ================ Flatten ================
        batch_size = encoder_outputs[0].shape[0]
        encoder_outputs = [torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs]

        # ================ Project ================
        projected = [self.projectors[i](eo) for i, eo in enumerate(encoder_outputs)]
        hidden = torch.stack(projected)
        hidden = torch.permute(hidden, (1, 0, 2))  # shape [bs, num_eo, h]

        # ================ Aggregate ================
        hidden = torch.mean(hidden, dim=1)

        # ================ Fully Connected ================
        if self.fc_stack is not None:
            hidden = self.fc_stack(hidden)

        return_data = {"combiner_output": hidden}

        if len(inputs) == 1:
            # Workaround for including additional tensors from output of input encoders for
            # potential use in decoders, e.g. LSTM state for seq2seq.
            # TODO(Justin): Think about how to make this communication work for multi-sequence
            # features. Other combiners.
            for key, value in [d for d in inputs.values()][0].items():
                if key != ENCODER_OUTPUT:
                    return_data[key] = value

        return return_data
