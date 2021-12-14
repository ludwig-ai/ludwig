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

import torch

from ludwig.modules.attention_modules import FeedForwardAttentionReducer
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule, sequence_length_3D

logger = logging.getLogger(__name__)


class SequenceReducer(LudwigModule):
    """Reduces the sequence dimension of an input tensor according to the specified reduce_mode.  Any additional
    kwargs are passed on to the reduce mode's constructor.  If using reduce_mode=="attention", the input_size kwarg
    must also be specified.

    A sequence is a tensor of 2 or more dimensions, where the shape is [batch size x sequence length x ...].

    :param reduce_mode: The reduction mode, one of {"last", "sum", "mean", "max", "concat", "attention", "none"}
    :param max_sequence_length The maximum sequence length.
    :param embedding_size The size of each sequence element/embedding vector, or None if input is a sequence of scalars.
    """

    def __init__(self, reduce_mode: str = None, max_sequence_length: int = None, embedding_size: int = None, **kwargs):
        super().__init__()
        # save as private variable for debugging
        self._reduce_mode = reduce_mode
        self._max_sequence_length = max_sequence_length
        self._embedding_size = embedding_size
        # If embedding size specified and mode is attention, use embedding size as attention module input size
        # unless the input_size kwarg is provided.
        if reduce_mode == "attention" and embedding_size and "input_size" not in kwargs:
            kwargs["input_size"] = embedding_size
        # use registry to find required reduction function
        self._reduce_obj = get_from_registry(reduce_mode, reduce_mode_registry)(**kwargs)

    def forward(self, inputs, mask=None):
        """Forward pass of reducer.

        :param inputs: A tensor of 2 or more dimensions, where the shape is [batch size x sequence length x ...].
        :param mask: A mask tensor of 2 dimensions [batch size x sequence length].  Not yet implemented.

        :return: The input after applying the reduction operation to sequence dimension.
        """
        return self._reduce_obj(inputs, mask=mask)

    @property
    def input_shape(self) -> torch.Size:
        """Returns size of the input tensor without the batch dimension."""
        if self._max_sequence_length is None:
            raise RuntimeError(
                "SequenceReducer must be initialized with max_sequence_length to get input_shape or output_shape."
            )
        elif self._embedding_size is None:
            return torch.Size([self._max_sequence_length])
        else:
            return torch.Size([self._max_sequence_length, self._embedding_size])


class ReduceLast(torch.nn.Module):
    def forward(self, inputs, mask=None):
        # inputs: [batch_size, seq_size, hidden_size]
        batch_size = inputs.shape[0]
        # gather the correct outputs from the the RNN outputs (the outputs after sequence_length are all 0s)
        # todo: review for generality
        sequence_length = sequence_length_3D(inputs) - 1
        sequence_length[sequence_length < 0] = 0
        gathered = inputs[torch.arange(batch_size), sequence_length.type(torch.int64)]
        return gathered


class ReduceSum(torch.nn.Module):
    def forward(self, inputs, mask=None):
        return torch.sum(inputs, dim=1)


class ReduceMean(torch.nn.Module):
    def forward(self, inputs, mask=None):
        return torch.mean(inputs, dim=1)


class ReduceMax(torch.nn.Module):
    def forward(self, inputs, mask=None):
        return torch.amax(inputs, dim=1)


class ReduceConcat(torch.nn.Module):
    def forward(self, inputs, mask=None):
        if inputs.dim() > 2:
            return inputs.reshape(-1, inputs.shape[-1] * inputs.shape[-2])
        return inputs


class ReduceNone(torch.nn.Module):
    def forward(self, inputs, mask=None):
        return inputs


reduce_mode_registry = {
    "last": ReduceLast,
    "sum": ReduceSum,
    "mean": ReduceMean,
    "avg": ReduceMean,
    "max": ReduceMax,
    "concat": ReduceConcat,
    "attention": FeedForwardAttentionReducer,
    # TODO: Simplify this.
    "none": ReduceNone,
    "None": ReduceNone,
    None: ReduceNone,
}
