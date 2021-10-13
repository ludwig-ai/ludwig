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
import logging

from enum import Enum
from typing import List, Dict, Optional, Type, Union
from pydantic import BaseModel, confloat, PositiveInt, NonNegativeInt, NonNegativeFloat, create_model

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

from ludwig.constants import NUMERICAL, BINARY, TYPE, NAME
from ludwig.encoders.sequence_encoders import ParallelCNN
from ludwig.encoders.sequence_encoders import StackedCNN
from ludwig.encoders.sequence_encoders import StackedCNNRNN
from ludwig.encoders.sequence_encoders import StackedParallelCNN
from ludwig.encoders.sequence_encoders import StackedRNN
from ludwig.modules.attention_modules import TransformerStack
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.initializer_modules import initializers_registry
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.modules.tabnet_modules import TabNet
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D

logger = logging.getLogger(__name__)

# Declare/shortcut to parameter registries:
preset_weights_initializer_registry = list(initializers_registry.keys())
preset_bias_initializer_registry = list(initializers_registry.keys())
weights_regularizer_registry = ['l1', 'l2', 'l1_l2']
bias_regularizer_registry = ['l1', 'l2', 'l1_l2']
activity_regularizer_registry = ['l1', 'l2', 'l1_l2']
norm_registry = ['batch', 'layer']
activation_registry = ['relu']
reduce_output_registry = ['sum', 'mean', 'sqrt', 'concat']
# reduce_output_registry = ['sum', 'mean', 'sqrt', 'concat', 'null']


# Initializers accept presets or customized dicts (not JSON-validated):
#
# Note: instead of inheriting from Enum directly, define StringEnum so that
# conversions to string work normally in rest of codebase:
class StringEnum(str, Enum):
    pass
WeightsInitializerEnum = \
    StringEnum("WeightsInitializerEnum", \
        {k:k for k in preset_weights_initializer_registry if k != None})
WeightsInitializerType = Union[WeightsInitializerEnum, Dict]
# WeightsInitializerType = WeightsInitializerEnum
BiasInitializerEnum = \
    StringEnum("BiasInitializerEnum", \
        {k:k for k in preset_bias_initializer_registry if k != None})
BiasInitializerType = Union[BiasInitializerEnum, Dict]
# BiasInitializerType = BiasInitializerEnum

WeightsRegularizerType = \
    StringEnum("WeightsRegularizerEnum",
        {k:k for k in weights_regularizer_registry})
BiasRegularizerType = \
    StringEnum("BiasRegularizerEnum",
        {k:k for k in bias_regularizer_registry})
ActivityRegularizerType = \
    StringEnum("ActivityRegularizerEnum",
        {k:k for k in activity_regularizer_registry})
NormType = \
    StringEnum("NormEnum",
        {k:k for k in norm_registry})
ActivationType = \
    StringEnum("ActivationEnum",
        {k:k for k in activation_registry})
ReduceOutputType = \
    StringEnum("ReduceOutputEnum",
        {k:k for k in reduce_output_registry})

# optional_fields = list()
# if required:
#   fields[col_name] = (data_type, ...)
# else:
#   fields[col_name] = (data_type, None)
#   optional_fields.append(col_name)

# model = pydantic.create_model(name, **fields)

def insert_nonetype_on_optionals_schema_extra(params_cls):
    # Find all optional (non-required) fields in class:
    print('here')
    optionalParams = []
    print(params_cls.__fields__)
    print(params_cls.__config__.fields)
    for param_name, field in params_cls.__fields__.items():
        if not field.required:
            optionalParams.append((param_name, field))
    print(optionalParams)
    # Note: ONLY called after the json is generated:
    def schema_extra(schema, model):
        print('called')
        print(schema)
        for param, field in optionalParams:
            print(param)
            print(field)
            if 'Enum' in field.type_.__name__:
                path = field.type_.__name__
                original_type = schema["definitions"][path]["type"]
                schema["definitions"][path].update({"type": ["null", original_type]})
            else:
                original_type = schema["properties"][param]["type"]
                schema["properties"][param].update({"type": ["null", original_type]})
    params_cls.__config__.schema_extra = schema_extra
    print(params_cls.__config__.fields)

# model.__config__.schema_extra = schema_extra

# schema_json = model.schema_json()

class ConcatCombinerParams(BaseModel):
    fc_layers: Optional[List[Dict]] = None
    num_fc_layers: Optional[NonNegativeInt] = None
    fc_size: PositiveInt = 256
    use_bias: bool = True
    weights_initializer: WeightsInitializerType = 'glorot_uniform'
    bias_initializer: BiasInitializerType = 'zeros'
    weights_regularizer: Optional[WeightsRegularizerType] = None
    bias_regularizer: Optional[BiasRegularizerType] = None
    activity_regularizer: Optional[ActivityRegularizerType] = None
    norm: Optional[NormType] = None
    norm_params: Optional[Dict] = None
    activation: ActivationType = 'relu'
    dropout: confloat(ge=0.0, le=1.0) = 0.0
    flatten_inputs: bool = False
    residual: bool = False

class ConcatCombiner(tf.keras.Model):
    def __init__(
            self,
            input_features: Optional[List] = None,
            config_params: ConcatCombinerParams = ConcatCombinerParams(),
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.flatten_inputs = config_params.flatten_inputs
        self.fc_stack = None

        # todo future: this may be redundant, check
        if config_params.fc_layers is None and \
                config_params.num_fc_layers is not None:
            fc_layers = []
            for i in range(config_params.num_fc_layers):
                fc_layers.append({'fc_size': config_params.fc_size})

        if config_params.fc_layers is not None:
            logger.debug('  FCStack')
            self.fc_stack = FCStack(
                layers=config_params.fc_layers,
                num_layers=config_params.num_fc_layers,
                default_fc_size=config_params.fc_size,
                default_use_bias=config_params.use_bias,
                default_weights_initializer=config_params.weights_initializer,
                default_bias_initializer=config_params.bias_initializer,
                default_weights_regularizer=config_params.weights_regularizer,
                default_bias_regularizer=config_params.bias_regularizer,
                default_activity_regularizer=config_params.activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=config_params.norm,
                default_norm_params=config_params.norm_params,
                default_activation=config_params.activation,
                default_dropout=config_params.dropout,
                residual=config_params.residual,
            )

        if input_features and len(input_features) == 1 and config_params.fc_layers is None:
            self.supports_masking = True

    def call(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]

        # ================ Flatten ================
        if self.flatten_inputs:
            batch_size = tf.shape(encoder_outputs[0])[0]
            encoder_outputs = [
                tf.reshape(eo, [batch_size, -1]) for eo in encoder_outputs
            ]

        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = concatenate(encoder_outputs, -1)
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

    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return ConcatCombinerParams


class SequenceConcatCombinerParams(BaseModel):
    reduce_output: Optional[ReduceOutputType] = None
    main_sequence_feature: Optional[str] = None

class SequenceConcatCombiner(tf.keras.Model):
    def __init__(
            self,
            config_params: SequenceConcatCombinerParams = SequenceConcatCombinerParams(),
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = config_params.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=config_params.reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True
        self.main_sequence_feature = config_params.main_sequence_feature

    def __call__(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
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
                    multipliers = tf.constant([1, sequence_max_length, 1])
                    tiled_representation = tf.tile(
                        tf.expand_dims(if_representation, 1),
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

        hidden = tf.concat(representations, 2)
        logger.debug('  concat_hidden: {0}'.format(hidden))

        # ================ Mask ================
        # todo future: maybe modify this with TF2 mask mechanics
        sequence_mask = tf.sequence_mask(
            sequence_length,
            sequence_max_length
        )
        hidden = tf.multiply(
            hidden,
            tf.cast(tf.expand_dims(sequence_mask, -1), dtype=tf.float32)
        )

        # ================ Reduce ================
        hidden = self.reduce_sequence(hidden)

        return_data = {'combiner_output': hidden}

        if len(inputs) == 1:
            for key, value in [d for d in inputs.values()][0].items():
                if key != 'encoder_output':
                    return_data[key] = value

        return return_data

    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return SequenceConcatCombinerParams


class SequenceCombinerParams(BaseModel):
    reduce_output: Optional[ReduceOutputType] = None
    main_sequence_feature: Optional[str] = None
    encoder: Optional[str] = None

class SequenceCombiner(tf.keras.Model):
    def __init__(
            self,
            config_params: SequenceCombinerParams = SequenceCombinerParams(),
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.combiner = SequenceConcatCombiner(SequenceConcatCombinerParams(
            reduce_output=None,
            main_sequence_feature=config_params.main_sequence_feature
        ))

        self.encoder_obj = get_from_registry(
            config_params.encoder, sequence_encoder_registry)(
            should_embed=False,
            reduce_output=config_params.reduce_output,
            **kwargs
        )

        if (hasattr(self.encoder_obj, 'supports_masking') and
                self.encoder_obj.supports_masking):
            self.supports_masking = True

    def __call__(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
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

    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return SequenceCombinerParams


class TabNetCombinerParams(BaseModel):
        size: PositiveInt = 32 # N_a in the paper
        output_size: PositiveInt = 32  # N_d in the paper
        num_steps: PositiveInt = 1  # N_steps in the paper
        num_total_blocks: PositiveInt = 4
        num_shared_blocks: PositiveInt = 2
        relaxation_factor: NonNegativeFloat = 1.5  # gamma in the paper
        bn_epsilon: float = 1e-3
        bn_momentum: float = 0.7  # m_B in the paper
        bn_virtual_bs: Optional[PositiveInt] = None  # B_v from the paper
        sparsity: float = 1e-5  # lambda_sparse in the paper
        dropout: confloat(ge=0.0, le=1.0) = 0

class TabNetCombiner(tf.keras.Model):
    def __init__(
            self,
            config_params: TabNetCombinerParams = TabNetCombinerParams(),
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.tabnet = TabNet(
            size=config_params.size,
            output_size=config_params.output_size,
            num_steps=config_params.num_steps,
            num_total_blocks=config_params.num_total_blocks,
            num_shared_blocks=config_params.num_shared_blocks,
            relaxation_factor=config_params.relaxation_factor,
            bn_epsilon=config_params.bn_epsilon,
            bn_momentum=config_params.bn_momentum,
            bn_virtual_bs=config_params.bn_virtual_bs,
            sparsity=config_params.sparsity
        )

        if config_params.dropout > 0:
            self.dropout = tf.keras.layers.Dropout(config_params.dropout)
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
    
    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return TabNetCombinerParams


class TransformerCombinerParams(BaseModel):
        num_layers: PositiveInt = 1
        hidden_size: PositiveInt = 256
        num_heads: PositiveInt = 8
        transformer_fc_size: PositiveInt = 256
        dropout: confloat(ge=0.0, le=1.0) = 0.1
        fc_layers: Optional[List[Dict]] = None
        num_fc_layers: NonNegativeInt = 0
        fc_size: PositiveInt = 256
        use_bias: bool = True
        weights_initializer: WeightsInitializerType = 'glorot_uniform'
        bias_initializer: BiasInitializerType ='zeros'
        weights_regularizer: Optional[WeightsRegularizerType] = None
        bias_regularizer: Optional[BiasRegularizerType] = None
        activity_regularizer: Optional[ActivityRegularizerType] = None
        # weights_constraint=None
        # bias_constraint=None
        norm: Optional[NormType] = None
        norm_params: Optional[Dict] = None
        fc_activation: ActivationType = 'relu'
        fc_dropout: confloat(ge=0.0, le=1.0) = 0
        fc_residual: bool = False
        # TODO: Note
        reduce_output: Optional[ReduceOutputType] = 'mean'

class TransformerCombiner(tf.keras.Model):
    def __init__(
            self,
            input_features: Optional[List] = None,
            config_params: TransformerCombinerParams = TransformerCombinerParams(),
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = config_params.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=config_params.reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True

        logger.debug('  Projectors')
        self.projectors = [Dense(config_params.hidden_size) for _ in input_features]

        logger.debug('  TransformerStack')
        self.transformer_stack = TransformerStack(
            hidden_size=config_params.hidden_size,
            num_heads=config_params.num_heads,
            fc_size=config_params.transformer_fc_size,
            num_layers=config_params.num_layers,
            dropout=config_params.dropout
        )

        if self.reduce_output is not None:
            logger.debug('  FCStack')
            self.fc_stack = FCStack(
                layers=config_params.fc_layers,
                num_layers=config_params.num_fc_layers,
                default_fc_size=config_params.fc_size,
                default_use_bias=config_params.use_bias,
                default_weights_initializer=config_params.weights_initializer,
                default_bias_initializer=config_params.bias_initializer,
                default_weights_regularizer=config_params.weights_regularizer,
                default_bias_regularizer=config_params.bias_regularizer,
                default_activity_regularizer=config_params.activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=config_params.norm,
                default_norm_params=config_params.norm_params,
                default_activation=config_params.fc_activation,
                default_dropout=config_params.fc_dropout,
                fc_residual=config_params.fc_residual,
            )

    def call(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]

        # ================ Flatten ================
        batch_size = tf.shape(encoder_outputs[0])[0]
        encoder_outputs = [
            tf.reshape(eo, [batch_size, -1]) for eo in encoder_outputs
        ]

        # ================ Project & Concat ================
        projected = [
            self.projectors[i](eo)
            for i, eo in enumerate(encoder_outputs)
        ]
        hidden = tf.stack(projected)  # num_eo, bs, h
        hidden = tf.transpose(hidden, perm=[1, 0, 2])  # bs, num_eo, h

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

    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return TransformerCombinerParams

class TabTransformerCombinerParams(BaseModel):
        embed_input_feature_name: Optional[Union[int, str]] = None  # None or embedding size or "add"
        num_layers: PositiveInt = 1
        hidden_size: PositiveInt = 256
        num_heads: PositiveInt = 8
        transformer_fc_size: PositiveInt = 256
        dropout: confloat(ge=0.0, le=1.0) = 0.1
        fc_layers: Optional[List[Dict]] = None
        num_fc_layers: NonNegativeInt = 0
        fc_size: PositiveInt = 256
        use_bias: bool = True
        weights_initializer: WeightsInitializerType = 'glorot_uniform'
        bias_initializer: BiasInitializerType ='zeros'
        weights_regularizer: Optional[WeightsRegularizerType] = None
        bias_regularizer: Optional[BiasRegularizerType] = None
        activity_regularizer: Optional[ActivityRegularizerType] = None
        # weights_constraint=None
        # bias_constraint=None
        norm: Optional[NormType] = None
        norm_params: Optional[Dict] = None
        fc_activation: ActivationType = 'relu'
        fc_dropout: confloat(ge=0.0, le=1.0) = 0
        fc_residual: bool = False
        # TODO: Note
        reduce_output: Optional[ReduceOutputType] = 'concat'

class TabTransformerCombiner(tf.keras.Model):
    def __init__(
            self,
            input_features: Optional[List] = None,
            config_params: TabTransformerCombinerParams = TabTransformerCombinerParams(),
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        if config_params.reduce_output is None:
            raise ValueError("TabTransformer requires the `resude_output` "
                             "parametr")
        self.reduce_output = config_params.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=config_params.reduce_output)
        self.supports_masking = True
        self.layer_norm = LayerNormalization()

        self.embed_input_feature_name = config_params.embed_input_feature_name
        if self.embed_input_feature_name:
            vocab = [i_f for i_f in input_features
                     if i_f[TYPE] != NUMERICAL or i_f[TYPE] != BINARY]
            if self.embed_input_feature_name == 'add':
                self.embed_i_f_name_layer = Embed(vocab, config_params.hidden_size,
                                                  force_embedding_size=True)
                projector_size = config_params.hidden_size
            elif isinstance(self.embed_input_feature_name, int):
                if self.embed_input_feature_name > config_params.hidden_size:
                    raise ValueError(
                        "TabTransformer parameter "
                        "`embed_input_feature_name` "
                        "specified integer value ({}) "
                        "needs to be smaller than "
                        "`hidden_size` ({}).".format(
                            self.embed_input_feature_name, config_params.hidden_size
                        ))
                self.embed_i_f_name_layer = Embed(
                    vocab,
                    self.embed_input_feature_name,
                    force_embedding_size=True,
                )
                projector_size = config_params.hidden_size - self.embed_input_feature_name
            else:
                raise ValueError("TabTransformer parameter "
                                 "`embed_input_feature_name` "
                                 "should be either None, an integer or `add`, "
                                 "the current value is "
                                 "{}".format(self.embed_input_feature_name))
        else:
            projector_size = config_params.hidden_size

        logger.debug('  Projectors')
        self.projectors = [Dense(projector_size) for i_f in input_features
                           if i_f[TYPE] != NUMERICAL and i_f[TYPE] != BINARY]
        self.skip_features = [i_f[NAME] for i_f in input_features
                              if i_f[TYPE] == NUMERICAL or i_f[TYPE] == BINARY]

        logger.debug('  TransformerStack')
        self.transformer_stack = TransformerStack(
            hidden_size=config_params.hidden_size,
            num_heads=config_params.num_heads,
            fc_size=config_params.transformer_fc_size,
            num_layers=config_params.num_layers,
            dropout=config_params.dropout
        )

        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            layers=config_params.fc_layers,
            num_layers=config_params.num_fc_layers,
            default_fc_size=config_params.fc_size,
            default_use_bias=config_params.use_bias,
            default_weights_initializer=config_params.weights_initializer,
            default_bias_initializer=config_params.bias_initializer,
            default_weights_regularizer=config_params.weights_regularizer,
            default_bias_regularizer=config_params.bias_regularizer,
            default_activity_regularizer=config_params.activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=config_params.norm,
            default_norm_params=config_params.norm_params,
            default_activation=config_params.fc_activation,
            default_dropout=config_params.fc_dropout,
            fc_residual=config_params.fc_residual,
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
    
    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return TabTransformerCombinerParams


class ComparatorCombinerParams(BaseModel):
    num_fc_layers: NonNegativeInt = 1
    fc_size: PositiveInt = 256
    use_bias: bool = True
    weights_initializer: WeightsInitializerType = 'glorot_uniform'
    bias_initializer: BiasInitializerType = 'zeros'
    weights_regularizer: Optional[WeightsRegularizerType] = None
    bias_regularizer: Optional[BiasRegularizerType] = None
    activity_regularizer: Optional[ActivityRegularizerType] = None
    # weights_constraint=None
    # bias_constraint=None
    norm: Optional[NormType] = None
    norm_params: Optional[Dict] = None
    activation: ActivationType = 'relu'
    dropout: confloat(ge=0.0, le=1.0) = 0


class ComparatorCombiner(tf.keras.Model):
    def __init__(
            self,
            entity_1: Optional[List[str]] = None,
            entity_2: Optional[List[str]] = None, 
            config_params: ComparatorCombinerParams = ComparatorCombinerParams(),
            **kwargs,
    ):
        super().__init__()
        logger.debug(" {}".format(self.name))

        self.fc_stack = None

        # todo future: this may be redundant, check
        # if fc_layers is None and num_fc_layers is not None:
        fc_layers = []
        for i in range(config_params.num_fc_layers):
            fc_layers.append({"fc_size": config_params.fc_size})

        if fc_layers is not None:
            logger.debug("  FCStack")
            self.e1_fc_stack = FCStack(
                layers=fc_layers,
                num_layers=config_params.num_fc_layers,
                default_fc_size=config_params.fc_size,
                default_use_bias=config_params.use_bias,
                default_weights_initializer=config_params.weights_initializer,
                default_bias_initializer=config_params.bias_initializer,
                default_weights_regularizer=config_params.weights_regularizer,
                default_bias_regularizer=config_params.bias_regularizer,
                default_activity_regularizer=config_params.activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=config_params.norm,
                default_norm_params=config_params.norm_params,
                default_activation=config_params.activation,
                default_dropout=config_params.dropout,
            )
            self.e2_fc_stack = FCStack(
                layers=fc_layers,
                num_layers=config_params.num_fc_layers,
                default_fc_size=config_params.fc_size,
                default_use_bias=config_params.use_bias,
                default_weights_initializer=config_params.weights_initializer,
                default_bias_initializer=config_params.bias_initializer,
                default_weights_regularizer=config_params.weights_regularizer,
                default_bias_regularizer=config_params.bias_regularizer,
                default_activity_regularizer=config_params.activity_regularizer,
                # default_weights_constraint=weights_constraint,
                # default_bias_constraint=bias_constraint,
                default_norm=config_params.norm,
                default_norm_params=config_params.norm_params,
                default_activation=config_params.activation,
                default_dropout=config_params.dropout,
            )

        # todo: this should actually be the size of the last fc layer,
        #  not just fc_size
        # todo: set initializer and regularization
        self.bilinear_weights = tf.random.normal([config_params.fc_size, config_params.fc_size],
                                                 dtype=tf.float32)

        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.required_inputs = set(entity_1 + entity_2)
        self.fc_size = config_params.fc_size

    def call(self, inputs, training=None, mask=None,
             **kwargs):  # encoder outputs
        assert (
                inputs.keys() == self.required_inputs
        ), f"Missing inputs {self.required_inputs - set(inputs.keys())}"

        ############
        # Entity 1 #
        ############
        e1_enc_outputs = [inputs[k]["encoder_output"] for k in self.entity_1]

        # ================ Flatten ================
        batch_size = tf.shape(e1_enc_outputs[0])[0]
        e1_enc_outputs = [
            tf.reshape(eo, [batch_size, -1]) for eo in e1_enc_outputs
        ]

        # ================ Concat ================
        if len(e1_enc_outputs) > 1:
            e1_hidden = concatenate(e1_enc_outputs, 1)
        else:
            e1_hidden = list(e1_enc_outputs)[0]

        # ================ Fully Connected ================
        e1_hidden = self.e1_fc_stack(e1_hidden, training=training, mask=mask)

        ############
        # Entity 2 #
        ############
        e2_enc_outputs = [inputs[k]["encoder_output"] for k in self.entity_2]

        # ================ Flatten ================
        batch_size = tf.shape(e2_enc_outputs[0])[0]
        e2_enc_outputs = [
            tf.reshape(eo, [batch_size, -1]) for eo in e2_enc_outputs
        ]

        # ================ Concat ================
        if len(e2_enc_outputs) > 1:
            e2_hidden = concatenate(e2_enc_outputs, 1)
        else:
            e2_hidden = list(e2_enc_outputs)[0]

        # ================ Fully Connected ================
        e2_hidden = self.e2_fc_stack(e2_hidden, training=training, mask=mask)

        ###########
        # Compare #
        ###########
        if e1_hidden.shape != e2_hidden.shape:
            raise ValueError(
                f"Mismatching shapes among dimensions! "
                f"entity1 shape: {e1_hidden.shape.as_list()} "
                f"entity2 shape: {e2_hidden.shape.as_list()}"
            )

        dot_product = tf.matmul(e1_hidden, tf.transpose(e2_hidden))
        element_wise_mul = tf.math.multiply(e1_hidden, e2_hidden)
        abs_diff = tf.abs(e1_hidden - e2_hidden)
        bilinear_prod = tf.matmul(
            e1_hidden,
            tf.matmul(self.bilinear_weights, tf.transpose(e2_hidden))
        )
        hidden = concatenate(
            [dot_product, element_wise_mul, abs_diff, bilinear_prod], 1
        )

        return {"combiner_output": hidden}
    
    @staticmethod
    def get_params_cls() -> Type[BaseModel]:
        return ComparatorCombinerParams

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
