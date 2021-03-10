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
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import concatenate

from ludwig.encoders.sequence_encoders import ParallelCNN
from ludwig.encoders.sequence_encoders import StackedCNN
from ludwig.encoders.sequence_encoders import StackedCNNRNN
from ludwig.encoders.sequence_encoders import StackedParallelCNN
from ludwig.encoders.sequence_encoders import StackedRNN
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D

logger = logging.getLogger(__name__)


class ConcatCombiner(tf.keras.Model):
    def __init__(
            self,
            input_features=None,
            fc_layers=None,
            num_fc_layers=None,
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
            activation='relu',
            dropout=0,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.fc_stack = None

        # todo future: this may be redundant, check
        if fc_layers is None and \
                num_fc_layers is not None:
            fc_layers = []
            for i in range(num_fc_layers):
                fc_layers.append({'fc_size': fc_size})

        if fc_layers is not None:
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
                default_activation=activation,
                default_dropout=dropout,
            )

        if input_features and len(input_features) == 1 and fc_layers is None:
            self.supports_masking = True

    def call(
            self,
            inputs,  # encoder outputs
            training=None,
            mask=None,
            **kwargs
    ):
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]
        # ================ Concat ================
        if len(encoder_outputs) > 1:
            hidden = concatenate(encoder_outputs, 1)
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


class SequenceConcatCombiner(tf.keras.Model):
    def __init__(
            self,
            reduce_output=None,
            main_sequence_feature=None,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True
        self.main_sequence_feature = main_sequence_feature

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


class SequenceCombiner(tf.keras.Model):
    def __init__(
            self,
            reduce_output=None,
            main_sequence_feature=None,
            encoder=None,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.combiner = SequenceConcatCombiner(
            reduce_output=None,
            main_sequence_feature=main_sequence_feature
        )

        self.encoder_obj = get_from_registry(
            encoder, sequence_encoder_registry)(
            should_embed=False,
            reduce_output=reduce_output,
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


class ComparatorCombiner(tf.keras.Model):
    def __init__(
            self,
            entity_1: List[str],
            entity_2: List[str],
            fc_layers=None,
            num_fc_layers=1,
            fc_size=256,
            use_bias=True,
            weights_initializer="glorot_uniform",
            bias_initializer="zeros",
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm=None,
            norm_params=None,
            activation="relu",
            dropout=0,
            **kwargs,
    ):
        super().__init__()
        logger.debug(" {}".format(self.name))

        self.fc_stack = None

        # todo future: this may be redundant, check
        if fc_layers is None and num_fc_layers is not None:
            fc_layers = []
            for i in range(num_fc_layers):
                fc_layers.append({"fc_size": fc_size})

        if fc_layers is not None:
            logger.debug("  FCStack")
            self.e1_fc_stack = FCStack(
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

        # todo: this should actually be the size of the last fc layer,
        #  not just fc_size
        # todo: set initializer and regularization
        self.bilinear_weights = tf.random.normal([fc_size, fc_size],
                                                 dtype=tf.float32)

        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.required_inputs = set(entity_1 + entity_2)
        self.fc_size = fc_size

    def call(self, inputs, training=None, mask=None,
             **kwargs):  # encoder outputs
        assert (
                inputs.keys() == self.required_inputs
        ), f"Missing inputs {self.required_inputs - set(inputs.keys())}"

        ############
        # Entity 1 #
        ############
        e1_enc_outputs = [inputs[k]["encoder_output"] for k in self.entity_1]

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
                f"Mismatching shapes among dimensions! entity1 shape: {e1_hidden.shape.as_list()} entity2 shape: {e2_hidden.shape.as_list()}"
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


def get_combiner_class(combiner_type):
    return get_from_registry(
        combiner_type,
        combiner_registry
    )


combiner_registry = {
    'concat': ConcatCombiner,
    'sequence_concat': SequenceConcatCombiner,
    'sequence': SequenceCombiner,
    'comparator': ComparatorCombiner,
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
