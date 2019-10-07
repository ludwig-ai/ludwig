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

import tensorflow as tf

from ludwig.features.feature_utils import SEQUENCE_TYPES
from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.models.modules.sequence_encoders import CNNRNN
from ludwig.models.modules.sequence_encoders import ParallelCNN
from ludwig.models.modules.sequence_encoders import RNN
from ludwig.models.modules.sequence_encoders import StackedCNN
from ludwig.models.modules.sequence_encoders import StackedParallelCNN
from ludwig.utils.misc import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D


logger = logging.getLogger(__name__)


class ConcatCombiner:
    def __init__(
            self,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
            norm=None,
            activation='relu',
            dropout=False,
            initializer=None,
            regularize=True,
            **kwargs
    ):
        self.fc_stack = None

        if fc_layers is None and \
                num_fc_layers is not None:
            fc_layers = []
            for i in range(num_fc_layers):
                fc_layers.append({'fc_size': fc_size})

        if fc_layers is not None:
            self.fc_stack = FCStack(
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_fc_size=fc_size,
                default_norm=norm,
                default_activation=activation,
                default_dropout=dropout,
                default_initializer=initializer,
                default_regularize=regularize
            )

    def __call__(
            self,
            feature_encodings,
            regularizer,
            dropout_rate,
            is_training=True,
            **kwargs
    ):
        representations = []
        representations_size = 0
        for fe_name, fe_properties in feature_encodings.items():
            representations.append(fe_properties['representation'])
            representations_size += fe_properties['size']

        scope_name = 'concat_combiner'
        with tf.compat.v1.variable_scope(scope_name):
            # ================ Concat ================
            hidden = tf.concat(representations, 1)
            hidden_size = representations_size

            logger.debug('  concat_hidden: {0}'.format(hidden))

            # ================ Fully Connected ================
            if self.fc_stack is not None:
                hidden = self.fc_stack(
                    hidden,
                    hidden_size,
                    regularizer=regularizer,
                    dropout_rate=dropout_rate,
                    is_training=is_training
                )

                hidden_size = self.fc_stack.layers[-1]['fc_size']
                logger.debug('  final_hidden: {0}'.format(hidden))

            hidden = tf.identity(hidden, name=scope_name)

        return hidden, hidden_size


class SequenceConcatCombiner:
    def __init__(
            self,
            reduce_output=None,
            main_sequence_feature=None,
            **kwargs
    ):
        self.reduce_output = reduce_output
        self.main_sequence_feature = main_sequence_feature

    def __call__(
            self,
            feature_encodings,
            regularizer,
            dropout_rate,
            **kwargs
    ):
        if (self.main_sequence_feature is None or
                self.main_sequence_feature not in feature_encodings):
            for fe_name, fe_properties in feature_encodings.items():
                if fe_properties['type'] in SEQUENCE_TYPES:
                    self.main_sequence_feature = fe_name
                    break

        if self.main_sequence_feature is None:
            raise Exception(
                'No sequence feature available for sequence combiner'
            )

        main_sequence_feature_encoding = \
            feature_encodings[self.main_sequence_feature]

        representation = main_sequence_feature_encoding['representation']
        representations_size = representation.shape[2].value
        representations = [representation]

        scope_name = 'sequence_concat_combiner'
        sequence_length = sequence_length_3D(representation)

        with tf.compat.v1.variable_scope(scope_name):
            # ================ Concat ================
            for fe_name, fe_properties in feature_encodings.items():
                if fe_name is not self.main_sequence_feature:
                    if fe_properties['type'] in SEQUENCE_TYPES and \
                            len(fe_properties['representation'].shape) == 3:
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
                        if fe_properties['representation'].shape[1] != \
                                representation.shape[1]:

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
                                    fe_properties['name'],
                                    fe_properties['representation'].shape[1],
                                    self.main_sequence_feature,
                                    representations_size,
                                    fe_properties['name'],
                                    fe_properties['representation'].shape,
                                    fe_properties['name'],
                                    representation.shape,
                                    fe_properties['name']
                                )
                            )
                        # this assumes all sequence representations have the
                        # same sequence length, 2nd dimension
                        representations.append(fe_properties['representation'])

                    elif len(fe_properties['representation'].shape) == 2:
                        sequence_max_length = tf.shape(representation)[1]
                        multipliers = tf.concat(
                            [[1], tf.expand_dims(sequence_max_length, -1), [1]],
                            0
                        )
                        tiled_representation = tf.tile(
                            tf.expand_dims(fe_properties['representation'], 1),
                            multipliers
                        )
                        logger.debug('  tiled_representation: {0}'.format(
                            tiled_representation))

                        mask = tf.sequence_mask(
                            sequence_length,
                            sequence_max_length
                        )
                        tiled_representation = tf.multiply(
                            tiled_representation,
                            tf.cast(tf.expand_dims(mask, -1), dtype=tf.float32)
                        )

                        representations.append(tiled_representation)

                    else:
                        raise ValueError(
                            'The representation of {} has rank {} and cannot be'
                            ' concatenated by a sequence concat combiner. '
                            'Only rank 2 and rank 3 tensors are supported.'.format(
                                fe_properties['name'],
                                len(fe_properties['representation'].shape)
                            )
                        )

                    representations_size += fe_properties['size']

            hidden = tf.concat(representations, 2)
            logger.debug('  concat_hidden: {0}'.format(hidden))
            hidden_size = representations_size

            # ================ Mask ================
            mask_matrix = tf.cast(
                tf.sign(
                    tf.reduce_sum(tf.abs(representation), -1, keep_dims=True)
                ),
                dtype=tf.float32
            )
            hidden = tf.multiply(hidden, mask_matrix)

            # ================ Reduce ================
            hidden = reduce_sequence(
                hidden,
                self.reduce_output
            )
            logger.debug('  reduced_concat_hidden: {0}'.format(hidden))

            hidden = tf.identity(hidden, name=scope_name)

        return hidden, hidden_size


class SequenceCombiner:
    def __init__(
            self,
            reduce_output=None,
            main_sequence_feature=None,
            encoder=None,
            **kwargs
    ):
        self.combiner = SequenceConcatCombiner(
            reduce_output=reduce_output,
            main_sequence_feature=main_sequence_feature
        )

        self.encoder_obj = get_from_registry(
            encoder, sequence_encoder_registry)(
            should_embed=False,
            **kwargs
        )

    def __call__(
            self,
            feature_encodings,
            regularizer,
            dropout_rate,
            is_training=True,
            **kwargs
    ):
        scope_name = 'sequence_combiner'
        with tf.compat.v1.variable_scope(scope_name):
            # ================ Concat ================
            hidden, hidden_size = self.combiner(
                feature_encodings,
                regularizer,
                dropout_rate,
                **kwargs
            )

            # ================ Sequence encoding ================
            hidden, hidden_size = self.encoder_obj(
                input_sequence=hidden,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
            logger.debug('  sequence_hidden: {0}'.format(hidden))

            hidden = tf.identity(hidden, name=scope_name)

        return hidden, hidden_size


def get_build_combiner(combiner_type):
    return get_from_registry(
        combiner_type,
        combiner_registry
    )


combiner_registry = {
    'concat': ConcatCombiner,
    'sequence_concat': SequenceConcatCombiner,
    'sequence': SequenceCombiner
}

sequence_encoder_registry = {
    'stacked_cnn': StackedCNN,
    'parallel_cnn': ParallelCNN,
    'stacked_parallel_cnn': StackedParallelCNN,
    'rnn': RNN,
    'cnnrnn': CNNRNN
}
