#! /usr/bin/env python
# coding=utf-8
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
from abc import abstractmethod
import logging
from typing import Union, Optional, List, Dict
from functools import lru_cache

import torch
from torch.nn import Module, ModuleList, Linear
from ludwig.utils.torch_utils import LudwigModule, \
    sequence_mask as torch_sequence_mask

from ludwig.constants import NUMERICAL, BINARY, TYPE, NAME
from ludwig.encoders.sequence_encoders import ParallelCNN
from ludwig.encoders.sequence_encoders import StackedCNN
from ludwig.encoders.sequence_encoders import StackedCNNRNN
from ludwig.encoders.sequence_encoders import StackedParallelCNN
from ludwig.encoders.sequence_encoders import StackedRNN
from ludwig.modules.attention_modules import TransformerStack
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.modules.tabnet_modules import TabNet
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import sequence_length_3D

logger = logging.getLogger(__name__)


# super class to house common properties
class CombinerClass(LudwigModule):
    @property
    @abstractmethod
    def concatenated_shape(self) -> torch.Size:
        """ Returns the size of concatenated encoder output tensors. """
        raise NotImplementedError('Abstract class.')

    @property
    def input_shape(self) -> Dict:
        # input to combiner is a dictionary of the input features encoder
        # outputs, this property returns dictionary of output shapes for each
        # input feature's encoder output shapes.
        return {k: self.input_features[k].output_shape
                for k in self.input_features}

    @property
    @lru_cache(maxsize=1)
    def output_shape(self) -> torch.Size:
        psuedo_input = {}
        for k in self.input_features:
            psuedo_input[k] = {
                'encoder_output': torch.rand(
                    2, *self.input_features[k].output_shape,
                    dtype=self.input_dtype
                )
            }
        output_tensor = self.forward(psuedo_input)
        return output_tensor['combiner_output'].size()[1:]


class ConcatCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict = None,
            fc_layers: Union[list, None] = None,
            num_fc_layers: Optional[int] = None,
            fc_size: int = 256,
            use_bias: bool = True,
            weights_initializer: str = 'xavier_uniform',
            bias_initializer: str = 'zeros',
            weights_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            activity_regularizer: Optional[str] = None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm: Optional[str] = None,
            norm_params: Optional[Dict] = None,
            activation: str = 'relu',
            dropout: float = 0,
            flatten_inputs: bool = False,
            residual: bool = False,
            **kwargs
    ):
        super().__init__()
        self.name = "ConcatCombiner"
        self.input_features = input_features
        logger.debug(' {}'.format(self.name))

        self.flatten_inputs = flatten_inputs
        self.fc_stack = None

        # todo future: this may be redundant, check
        if fc_layers is None and \
                num_fc_layers is not None:
            fc_layers = []
            for i in range(num_fc_layers):
                fc_layers.append({'fc_size': fc_size})

        self.fc_layers = fc_layers

        if fc_layers is not None:
            logger.debug('  FCStack')
            self.fc_stack = FCStack(
                first_layer_input_size=self.concatenated_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_fc_size=fc_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_weights_regularizer=weights_regularizer,
                default_bias_regularizer=bias_regularizer,
                default_activity_regularizer=activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
                residual=residual,
            )

        if input_features and len(input_features) == 1 and fc_layers is None:
            self.supports_masking = True

    @property
    def concatenated_shape(self) -> torch.Size:
        # compute the size of the last dimension for the incoming encoder outputs
        # this is required to setup the fully connected layer
        shapes = [
            torch.prod(torch.Tensor([*self.input_features[k].output_shape]))
            for k in self.input_features]
        return torch.Size([torch.sum(torch.Tensor(shapes)).type(torch.int32)])

    def forward(
            self,
            inputs: Dict,  # encoder outputs
            training: Optional[bool] = None,
            mask: Optional[bool] = None,
            **kwargs
    ):
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]

        # ================ Flatten ================
        if self.flatten_inputs:
            batch_size = encoder_outputs[0].shape[0]
            encoder_outputs = [
                torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs
            ]

        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = torch.cat(encoder_outputs, 1)
        else:
            hidden = list(encoder_outputs)[0]

        # ================ Fully Connected ================
        if self.fc_stack is not None:
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return_data = {'combiner_output': hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data


class SequenceConcatCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict,
            reduce_output: Optional[str] = None,
            main_sequence_feature: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        self.name = 'SequenceConcatCombiner'
        logger.debug(' {}'.format(self.name))

        self.input_features = input_features
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True
        self.main_sequence_feature = main_sequence_feature

    @property
    def concatenated_shape(self) -> torch.Size:
        # computes the effective shape of the input tensor after combining
        # all the encoder outputs
        # determine sequence size by finding the first sequence tensor
        # assume all the sequences are of the same size, if not true
        # this will be caught during processing
        seq_size = None
        for k in self.input_features:
            # dim-2 output_shape implies a sequence [seq_size, hidden]
            if len(self.input_features[k].output_shape) == 2:
                seq_size = self.input_features[k].output_shape[0]
                break

        # collect the size of the last dimension for all input feature
        # encoder outputs
        shapes = [self.input_features[k].output_shape[-1] for k in
                  self.input_features]  # output shape not input shape
        return torch.Size([seq_size, sum(shapes)])

    def forward(
            self,
            inputs: Dict,  # encoder outputs
            training: Optional[bool] = None,
            mask: Optional[bool] = None,
            **kwargs
    ):
        if (self.main_sequence_feature is None or
                self.main_sequence_feature not in inputs):
            for if_name, if_outputs in inputs.items():
                # todo: when https://github.com/ludwig-ai/ludwig/issues/810 is closed
                #       convert following test from using shape to use explicit
                #       if_outputs[TYPE] values for sequence features
                if len(if_outputs['encoder_output'].shape) == 3:
                    self.main_sequence_feature = if_name
                    break

        if self.main_sequence_feature is None:
            raise Exception(
                'No sequence feature available for sequence combiner'
            )

        main_sequence_feature_encoding = inputs[self.main_sequence_feature]

        representation = main_sequence_feature_encoding['encoder_output']
        representations = [representation]

        sequence_max_length = representation.shape[1]
        sequence_length = sequence_length_3D(representation)

        # ================ Concat ================
        for if_name, if_outputs in inputs.items():
            if if_name != self.main_sequence_feature:
                if_representation = if_outputs['encoder_output']
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
                            'The sequence length of the input feature {} '
                            'is {} and is different from the sequence '
                            'length of the main sequence feature {} which '
                            'is {}.\n Shape of {}: {}, shape of {}: {}.\n'
                            'Sequence lengths of all sequential features '
                            'must be the same  in order to be concatenated '
                            'by the sequence concat combiner. '
                            'Try to impose the same max sequence length '
                            'as a preprocessing parameter to both features '
                            'or to reduce the output of {}.'.format(
                                if_name,
                                if_representation.shape[1],
                                self.main_sequence_feature,
                                representation.shape[1],
                                if_name,
                                if_representation.shape,
                                if_name,
                                representation.shape,
                                if_name
                            )
                        )
                    # this assumes all sequence representations have the
                    # same sequence length, 2nd dimension
                    representations.append(if_representation)

                elif len(if_representation.shape) == 2:
                    multipliers = (1, sequence_max_length, 1)
                    tiled_representation = torch.tile(
                        torch.unsqueeze(if_representation, 1),
                        multipliers
                    )
                    representations.append(tiled_representation)

                else:
                    raise ValueError(
                        'The representation of {} has rank {} and cannot be'
                        ' concatenated by a sequence concat combiner. '
                        'Only rank 2 and rank 3 tensors are supported.'.format(
                            if_outputs['name'],
                            len(if_representation.shape)
                        )
                    )

        hidden = torch.cat(representations, 2)
        logger.debug('  concat_hidden: {0}'.format(hidden))

        # ================ Mask ================
        # todo future: maybe modify this with TF2 mask mechanics
        sequence_mask = torch_sequence_mask(
            sequence_length,
            sequence_max_length
        )
        hidden = torch.multiply(
            hidden,
            torch.unsqueeze(sequence_mask, -1).type(torch.float32)
        )

        # ================ Reduce ================
        hidden = self.reduce_sequence(hidden)

        return_data = {'combiner_output': hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data


class SequenceCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict,
            reduce_output: Optional[str] = None,
            main_sequence_feature: Optional[str] = None,
            encoder: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        self.name = 'SequenceCombiner'
        logger.debug(' {}'.format(self.name))

        self.input_features = input_features

        self.combiner = SequenceConcatCombiner(
            input_features,
            reduce_output=None,
            main_sequence_feature=main_sequence_feature
        )

        logger.debug(
            f'combiner input shape {self.combiner.concatenated_shape}, '
            f'output shape {self.combiner.output_shape}'
        )

        self.encoder_obj = get_from_registry(
            encoder, sequence_encoder_registry)(
            should_embed=False,
            reduce_output=reduce_output,
            embedding_size=self.combiner.output_shape[1],
            max_sequence_length=self.combiner.output_shape[0],
            **kwargs
        )

        if (hasattr(self.encoder_obj, 'supports_masking') and
                self.encoder_obj.supports_masking):
            self.supports_masking = True

    @property
    def concatenated_shape(self) -> torch.Size:
        # computes the effective shape of the input tensor after combining
        # all the encoder outputs
        # determine sequence size by finding the first sequence tensor
        # assume all the sequences are of the same size, if not true
        # this will be caught during processing
        seq_size = None
        for k in self.input_features:
            # dim-2 output_shape implies a sequence [seq_size, hidden]
            if len(self.input_features[k].output_shape) == 2:
                seq_size = self.input_features[k].output_shape[0]
                break

        # collect the size of the last dimension for all input feature
        # encoder outputs
        shapes = [self.input_features[k].output_shape[-1] for k in
                  self.input_features]  # output shape not input shape
        return torch.Size([seq_size, sum(shapes)])

    def forward(
            self,
            inputs,  # encoder outputs
            training: Optional[bool] = None,
            mask: Optional[bool] = None,
            **kwargs
    ):
        # ================ Concat ================
        hidden = self.combiner(
            inputs,  # encoder outputs
            training=training,
            **kwargs
        )

        # ================ Sequence encoding ================
        hidden = self.encoder_obj(
            hidden['combiner_output'],
            training=training,
            **kwargs
        )

        return_data = {'combiner_output': hidden['encoder_output']}
        for key, value in hidden.items():
            if key != 'encoder_output':
                return_data[key] = value

        return return_data


class TabNetCombiner(Module):
    def __init__(
            self,
            size: int,  # N_a in the paper
            output_size: int,  # N_d in the paper
            num_steps: int = 1,  # N_steps in the paper
            num_total_blocks: int = 4,
            num_shared_blocks: int = 2,
            relaxation_factor: float = 1.5,  # gamma in the paper
            bn_epsilon: float = 1e-3,
            bn_momentum: float = 0.7,  # m_B in the paper
            bn_virtual_bs: int = None,  # B_v from the paper
            sparsity: float = 1e-5,  # lambda_sparse in the paper
            dropout=0,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.tabnet = TabNet(
            size=size,
            output_size=output_size,
            num_steps=num_steps,
            num_total_blocks=num_total_blocks,
            num_shared_blocks=num_shared_blocks,
            relaxation_factor=relaxation_factor,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
            bn_virtual_bs=bn_virtual_bs,
            sparsity=sparsity
        )

        if dropout > 0:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None

    def build(self, input_shape):
        self.flatten_layers = {
            k: tf.keras.layers.Flatten()
            for k in input_shape.keys()
        }

    def call(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        encoder_output_map = {
            k: inputs[k]['encoder_output'] for k in inputs
        }

        # ================ Flatten ================
        encoder_outputs = [
            self.flatten_layers[k](eo)
            for k, eo in encoder_output_map.items()
        ]

        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = concatenate(encoder_outputs, 1)
        else:
            hidden = list(encoder_outputs)[0]

        # ================ TabNet ================
        hidden, aggregated_mask, masks = self.tabnet(
            hidden,
            training=training,
        )
        if self.dropout:
            hidden = self.dropout(hidden, training=training)

        return_data = {'combiner_output': hidden,
                       'aggregated_attention_masks': aggregated_mask,
                       'attention_masks': masks}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data


class TransformerCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict = None,
            num_layers: int = 1,
            hidden_size: int = 256,
            num_heads: int = 8,
            transformer_fc_size: int = 256,
            dropout: float = 0.1,
            fc_layers: Optional[list] = None,
            num_fc_layers: int = 0,
            fc_size: int = 256,
            use_bias: bool = True,
            weights_initializer: str = 'xavier_uniform',
            bias_initializer: str = 'zeros',
            weights_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            activity_regularizer: Optional[str] = None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm: Optional[str] = None,
            norm_params: Optional[Dict] = None,
            fc_activation: str = 'relu',
            fc_dropout: float = 0,
            fc_residual: bool = False,
            reduce_output: Optional[str] = 'mean',
            **kwargs
    ):
        super().__init__()
        self.name = 'TransformerCombiner'
        logger.debug(' {}'.format(self.name))

        self.input_features = input_features
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True

        # sequence size for Transformer layer is number of input features
        self.sequence_size = len(self.input_features)

        logger.debug('  Projectors')
        self.projectors = ModuleList(
            # regardless of rank-2 or rank-3 input, torch.prod() calculates size
            # after flattening the encoder output tensor
            [Linear(
                torch.prod(
                    torch.Tensor([*input_features[inp].output_shape])
                ).type(torch.int32), hidden_size) for inp in input_features
            ]
        )

        logger.debug('  TransformerStack')
        self.transformer_stack = TransformerStack(
            input_size=hidden_size,
            sequence_size=self.sequence_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            fc_size=transformer_fc_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if self.reduce_output is not None:
            logger.debug('  FCStack')
            self.fc_stack = FCStack(
                self.transformer_stack.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_fc_size=fc_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_weights_regularizer=weights_regularizer,
                default_bias_regularizer=bias_regularizer,
                default_activity_regularizer=activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=fc_activation,
                default_dropout=fc_dropout,
                fc_residual=fc_residual,
            )

    @property
    def concatenated_shape(self) -> torch.Size:
        # compute the size of the last dimension for the incoming encoder outputs
        # this is required to setup the fully connected layer
        shapes = [
            torch.prod(torch.Tensor([*self.input_features[k].output_shape]))
            for k in self.input_features]
        return torch.Size([torch.sum(torch.Tensor(shapes)).type(torch.int32)])

    def forward(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]

        # ================ Flatten ================
        batch_size = encoder_outputs[0].shape[0]
        encoder_outputs = [
            torch.reshape(eo, [batch_size, -1]) for eo in encoder_outputs
        ]

        # ================ Project & Concat ================
        projected = [
            self.projectors[i](eo)
            for i, eo in enumerate(encoder_outputs)
        ]
        hidden = torch.stack(projected)  # shape [num_eo, bs, h]
        hidden = torch.permute(hidden, (1, 0, 2))  # shape [bs, num_eo, h]

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return_data = {'combiner_output': hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data


class TabTransformerCombiner(Module):
    def __init__(
            self,
            input_features=None,
            embed_input_feature_name=None,  # None or embedding size or "add"
            num_layers=1,
            hidden_size=256,
            num_heads=8,
            transformer_fc_size=256,
            dropout=0.1,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=256,
            use_bias=True,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm=None,
            norm_params=None,
            fc_activation='relu',
            fc_dropout=0,
            fc_residual=False,
            reduce_output='concat',
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        if reduce_output is None:
            raise ValueError("TabTransformer requires the `resude_output` "
                             "parametr")
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.supports_masking = True
        self.layer_norm = LayerNormalization()

        self.embed_input_feature_name = embed_input_feature_name
        if self.embed_input_feature_name:
            vocab = [i_f for i_f in input_features
                     if i_f[TYPE] != NUMERICAL or i_f[TYPE] != BINARY]
            if self.embed_input_feature_name == 'add':
                self.embed_i_f_name_layer = Embed(vocab, hidden_size,
                                                  force_embedding_size=True)
                projector_size = hidden_size
            elif isinstance(self.embed_input_feature_name, int):
                if self.embed_input_feature_name > hidden_size:
                    raise ValueError(
                        "TabTransformer parameter "
                        "`embed_input_feature_name` "
                        "specified integer value ({}) "
                        "needs to be smaller than "
                        "`hidden_size` ({}).".format(
                            self.embed_input_feature_name, hidden_size
                        ))
                self.embed_i_f_name_layer = Embed(
                    vocab,
                    self.embed_input_feature_name,
                    force_embedding_size=True,
                )
                projector_size = hidden_size - self.embed_input_feature_name
            else:
                raise ValueError("TabTransformer parameter "
                                 "`embed_input_feature_name` "
                                 "should be either None, an integer or `add`, "
                                 "the current value is "
                                 "{}".format(self.embed_input_feature_name))
        else:
            projector_size = hidden_size

        logger.debug('  Projectors')
        self.projectors = [Dense(projector_size) for i_f in input_features
                           if i_f[TYPE] != NUMERICAL and i_f[TYPE] != BINARY]
        self.skip_features = [i_f[NAME] for i_f in input_features
                              if i_f[TYPE] == NUMERICAL or i_f[TYPE] == BINARY]

        logger.debug('  TransformerStack')
        self.transformer_stack = TransformerStack(
            hidden_size=hidden_size,
            num_heads=num_heads,
            fc_size=transformer_fc_size,
            num_layers=num_layers,
            dropout=dropout
        )

        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=fc_activation,
            default_dropout=fc_dropout,
            fc_residual=fc_residual,
        )

    def call(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        skip_encoder_outputs = [inputs[k]['encoder_output'] for k in inputs
                                if k in self.skip_features]
        other_encoder_outputs = [inputs[k]['encoder_output'] for k in inputs
                                 if k not in self.skip_features]

        # ================ Flatten ================
        batch_size = tf.shape(other_encoder_outputs[0])[0]
        other_encoder_outputs = [
            tf.reshape(eo, [batch_size, -1]) for eo in other_encoder_outputs
        ]
        skip_encoder_outputs = [
            tf.reshape(eo, [batch_size, -1]) for eo in skip_encoder_outputs
        ]

        # ================ Project & Concat others ================
        projected = [
            self.projectors[i](eo)
            for i, eo in enumerate(other_encoder_outputs)
        ]
        hidden = tf.stack(projected)  # num_eo, bs, h
        hidden = tf.transpose(hidden, perm=[1, 0, 2])  # bs, num_eo, h

        if self.embed_input_feature_name:
            i_f_names_idcs = tf.range(0, len(other_encoder_outputs))
            embedded_i_f_names = self.embed_i_f_name_layer(i_f_names_idcs)
            embedded_i_f_names = tf.expand_dims(embedded_i_f_names, axis=0)
            embedded_i_f_names = tf.tile(embedded_i_f_names, [batch_size, 1, 1])
            if self.embed_input_feature_name == 'add':
               hidden = hidden + embedded_i_f_names
            else:
                hidden = tf.concat([hidden, embedded_i_f_names], axis=-1)

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        hidden = self.reduce_sequence(hidden)

        # ================ Concat Skipped ================
        if len(skip_encoder_outputs) > 1:
            skip_hidden = concatenate(skip_encoder_outputs, -1)
        else:
            skip_hidden = list(skip_encoder_outputs)[0]
        skip_hidden = self.layer_norm(skip_hidden)

        # ================ Concat Skipped and Others ================
        hidden = concatenate([hidden, skip_hidden], -1)

        # ================ FC Layers ================
        hidden = self.fc_stack(
            hidden,
            training=training,
            mask=mask
        )

        return_data = {'combiner_output': hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data


class ComparatorCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict,
            entity_1: List[str],
            entity_2: List[str],
            fc_layers: Optional[list] = None,
            num_fc_layers: int = 1,
            fc_size: int = 256,
            use_bias: bool = True,
            weights_initializer: str = "xavier_uniform",
            bias_initializer: str = "zeros",
            weights_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            activity_regularizer: Optional[str] = None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm: Optional[str] = None,
            norm_params: Optional[Dict] = None,
            activation: str = "relu",
            dropout: float = 0,
            **kwargs,
    ):
        super().__init__()
        self.name = "ComparatorCombiner"
        logger.debug("Entering {}".format(self.name))

        self.input_features = input_features
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.required_inputs = set(entity_1 + entity_2)
        self.fc_size = fc_size

        self.fc_stack = None

        # todo future: this may be redundant, check
        if fc_layers is None and num_fc_layers is not None:
            fc_layers = []
            for i in range(num_fc_layers):
                fc_layers.append({"fc_size": fc_size})

        if fc_layers is not None:
            logger.debug("Setting up FCStack")
            self.e1_fc_stack = FCStack(
                self.get_entity_shape(entity_1)[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_fc_size=fc_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_weights_regularizer=weights_regularizer,
                default_bias_regularizer=bias_regularizer,
                default_activity_regularizer=activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
            )
            self.e2_fc_stack = FCStack(
                self.get_entity_shape(entity_2)[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_fc_size=fc_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_weights_regularizer=weights_regularizer,
                default_bias_regularizer=bias_regularizer,
                default_activity_regularizer=activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
            )

        self.last_fc_layer_fc_size = fc_layers[-1]['fc_size']

        # todo: set initializer and regularization
        self.bilinear_weights = torch.randn(
            [self.last_fc_layer_fc_size, self.last_fc_layer_fc_size],
            dtype=torch.float32
        )

    def get_entity_shape(self, entity: list) -> torch.Size:
        sizes = [
            torch.prod(torch.Tensor([*self.input_features[k].output_shape]))
            for k in entity]
        return torch.Size([torch.sum(torch.Tensor(sizes)).type(torch.int32)])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([2 * self.last_fc_layer_fc_size + 2])

    def forward(
            self,
            inputs: Dict,
            training: Optional[bool] = None,
            mask: Optional[bool] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:  # encoder outputs
        assert (
                inputs.keys() == self.required_inputs
        ), f"Missing inputs {self.required_inputs - set(inputs.keys())}"

        ############
        # Entity 1 #
        ############
        e1_enc_outputs = [inputs[k]["encoder_output"] for k in self.entity_1]

        # ================ Flatten ================
        batch_size = e1_enc_outputs[0].shape[0]
        e1_enc_outputs = [
            torch.reshape(eo, [batch_size, -1]) for eo in e1_enc_outputs
        ]

        # ================ Concat ================
        if len(e1_enc_outputs) > 1:
            e1_hidden = torch.cat(e1_enc_outputs, 1)
        else:
            e1_hidden = list(e1_enc_outputs)[0]

        # ================ Fully Connected ================
        e1_hidden = self.e1_fc_stack(e1_hidden, training=training,
                                     mask=mask)  # [bs, fc_size]

        ############
        # Entity 2 #
        ############
        e2_enc_outputs = [inputs[k]["encoder_output"] for k in self.entity_2]

        # ================ Flatten ================
        batch_size = e2_enc_outputs[0].shape[0]
        e2_enc_outputs = [
            torch.reshape(eo, [batch_size, -1]) for eo in e2_enc_outputs
        ]

        # ================ Concat ================
        if len(e2_enc_outputs) > 1:
            e2_hidden = torch.cat(e2_enc_outputs, 1)
        else:
            e2_hidden = list(e2_enc_outputs)[0]

        # ================ Fully Connected ================
        e2_hidden = self.e2_fc_stack(e2_hidden, training=training,
                                     mask=mask)  # [bs, fc_size]

        ###########
        # Compare #
        ###########
        if e1_hidden.shape != e2_hidden.shape:
            raise ValueError(
                f"Mismatching shapes among dimensions! "
                f"entity1 shape: {e1_hidden.shape} "
                f"entity2 shape: {e2_hidden.shape}"
            )

        element_wise_mul = e1_hidden * e2_hidden  # [bs, fc_size]
        dot_product = torch.sum(element_wise_mul, 1, keepdim=True)  # [bs, 1]
        abs_diff = torch.abs(e1_hidden - e2_hidden)  # [bs, fc_size]
        bilinear_prod = torch.sum(
            torch.mm(e1_hidden, self.bilinear_weights) * e2_hidden,
            # [bs, fc_size]
            1, keepdim=True
        )  # [bs, 1]

        logger.debug(
            'preparing combiner output by concatenating these tensors: '
            f'dot_product: {dot_product.shape}, element_size_mul: {element_wise_mul.shape}'
            f', abs_diff: {abs_diff.shape}, bilinear_prod {bilinear_prod.shape}'
        )
        hidden = torch.cat(
            [dot_product, element_wise_mul, abs_diff, bilinear_prod], 1
        )  # [bs, 2 * fc_size + 2]

        return {"combiner_output": hidden}


def get_combiner_class(combiner_type):
    return get_from_registry(
        combiner_type,
        combiner_registry
    )


combiner_registry = {
    'concat': ConcatCombiner,
    'sequence_concat': SequenceConcatCombiner,
    'sequence': SequenceCombiner,
    'tabnet': TabNetCombiner,
    'comparator': ComparatorCombiner,
    "transformer": TransformerCombiner,
    "tabtransformer": TabTransformerCombiner,
}

sequence_encoder_registry = {
    'stacked_cnn': StackedCNN,
    'parallel_cnn': ParallelCNN,
    'stacked_parallel_cnn': StackedParallelCNN,
    'rnn': StackedRNN,
    'cnnrnn': StackedCNNRNN,
    # todo: add transformer
    # 'transformer': StackedTransformer,
}
