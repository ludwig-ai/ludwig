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

from ludwig.models.modules.attention_modules import \
    reduce_feed_forward_attention
from ludwig.utils.misc import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D


logger = logging.getLogger(__name__)


def reduce_last(sequence, **kwargs):
    batch_size = tf.shape(sequence)[0]
    sequence_length = sequence_length_3D(sequence)
    # gather the correct outputs from the the RNN outputs (the outputs after sequence_length are all 0s)
    return tf.gather_nd(sequence, tf.stack(
        [tf.range(batch_size), tf.maximum(sequence_length - 1, 0)], axis=1))


def reduce_sum(sequence, **kwargs):
    return tf.reduce_sum(sequence, axis=1)


def reduce_mean(sequence, **kwargs):
    return tf.reduce_mean(sequence, axis=1)


def reduce_max(sequence, **kwargs):
    return tf.reduce_max(sequence, axis=1)


def reduce_concat(sequence, **kwargs):
    if sequence.shape.as_list()[-2] is None or sequence.shape.as_list()[
        -1] is None:
        # this the case of outputs coming from rnn encoders
        logger.warning('  WARNING: '
                        'The sequence length dimension is undefined '
                        '(probably because of an RNN based encoder), '
                        'so the sequence cannot be reduced by concatenation. '
                        'Last will be used instead.')
        return reduce_last(sequence, **kwargs)
    else:
        return tf.reshape(sequence,
                          [-1, sequence.shape[-2] * sequence.shape[-1]])


def dont_reduce(sequence, **kwargs):
    return sequence


reduce_mode_registry = {
    'last': reduce_last,
    'sum': reduce_sum,
    'mean': reduce_mean,
    'avg': reduce_mean,
    'max': reduce_max,
    'concat': reduce_concat,
    'attention': reduce_feed_forward_attention,
    'none': dont_reduce,
    'None': dont_reduce,
    None: dont_reduce
}


def reduce_sequence(sequence, mode):
    reduce_mode = get_from_registry(
        mode,
        reduce_mode_registry
    )
    return reduce_mode(sequence)


def reduce_sequence_list(sequence_list, mode):
    reduce_mode = get_from_registry(
        mode,
        reduce_mode_registry
    )
    reduced_list = []
    for sequence in sequence_list:
        reduced_list.append(reduce_mode(sequence))
    if len(reduced_list) > 1:
        if reduce_mode == dont_reduce:
            reduced_output = tf.concat(reduced_list, 2)
        else:
            reduced_output = tf.concat(reduced_list, 1)
    else:
        reduced_output = reduced_list[0]
    return reduced_output
