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

# import tensorflow as tf
# from tensorflow.keras.layers import Layer

import torch
# from torch.nn import Module

from ludwig.modules.attention_modules import FeedForwardAttentionReducer
from ludwig.utils.misc_utils import get_from_registry
#from ludwig.utils.tf_utils import sequence_length_3D
from ludwig.utils.torch_utils import sequence_length_3D, LudwigModule, LudwigComponent

logger = logging.getLogger(__name__)


class SequenceReducer(LudwigComponent):

    def __init__(self, reduce_mode=None):
        super().__init__()
        # save as private variable for debugging
        self._reduce_mode = reduce_mode

        # use registry to find required reduction function
        self._reduce_obj = get_from_registry(
            reduce_mode,
            reduce_mode_registry
        )()

    def forward(self, inputs, training=None, mask=None):
        return self._reduce_obj(inputs, training=training, mask=mask)


class ReduceLast(LudwigComponent):

    def forward(self, inputs, training=None, mask=None):
        #batch_size = tf.shape(inputs)[0]
        batch_size = inputs.shape[0]
        sequence_length = sequence_length_3D(inputs)
        # gather the correct outputs from the the RNN outputs (the outputs after sequence_length are all 0s)
        '''
        gathered = tf.gather_nd(
            inputs,
            tf.stack(
                [tf.range(batch_size), tf.maximum(sequence_length - 1, 0)],
                axis=1
            )
        )
        '''
        gathered = SOME_FUNC_HERE(
            inputs,
            torch.stack(
                [torch.range(batch_size), torch.maximum(sequence_length - 1, 0)],
                dim=1
            )
        )
        return gathered


class ReduceSum(LudwigComponent):

    def forward(self, inputs, training=None, mask=None):
        #return tf.reduce_sum(inputs, axis=1)
        return torch.sum(inputs, dim=1)


class ReduceMean(LudwigComponent):

    def forward(self, inputs, training=None, mask=None):
        #return tf.reduce_mean(inputs, axis=1)
        return torch.mean(inputs, dim=1)


class ReduceMax(LudwigComponent):

    def forward(self, inputs, training=None, mask=None):
        #return tf.reduce_max(inputs, axis=1)
        return torch.max(inputs, dim=1)


class ReduceConcat(LudwigComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reduce_last = ReduceLast()

    def forward(self, inputs, training=None, mask=None):
        '''
        if (inputs.shape.as_list()[-2] is None or
                inputs.shape.as_list()[-1] is None):
        '''
        if (list(inputs.shape)[-2] is None or
                list(inputs.shape.as_list()[-1] is None)):
            # this the case of outputs coming from rnn encoders
            logger.warning('  WARNING: '
                           'The sequence length dimension is undefined '
                           '(probably because of an RNN based encoder), '
                           'so the sequence cannot be reduced '
                           'by concatenation. '
                           'Last will be used instead.')
            return self.reduce_last(inputs)
        else:
            '''
            return tf.reshape(
                inputs,
                [-1, inputs.shape[-2] * inputs.shape[-1]]
            )
            '''
            return torch.reshape(
                inputs,
                (-1, inputs.shape[-2] * inputs.shape[-1])
            )



class ReduceNone(LudwigComponent):

    def forward(self, inputs, training=None, mask=None):
        return inputs


reduce_mode_registry = {
    'last': ReduceLast,
    'sum': ReduceSum,
    'mean': ReduceMean,
    'avg': ReduceMean,
    'max': ReduceMax,
    'concat': ReduceConcat,
    'attention': FeedForwardAttentionReducer,
    'none': ReduceNone,
    'None': ReduceNone,
    None: ReduceNone
}
