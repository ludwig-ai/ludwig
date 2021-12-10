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
from ludwig.utils.torch_utils import sequence_length_3D

logger = logging.getLogger(__name__)


class SequenceReducer(torch.nn.Module):
    def __init__(self, reduce_mode=None, **kwargs):
        super().__init__()
        # save as private variable for debugging
        self._reduce_mode = reduce_mode

        # use registry to find required reduction function
        self._reduce_obj = get_from_registry(reduce_mode, reduce_mode_registry)(**kwargs)

    def forward(self, inputs, mask=None):
        return self._reduce_obj(inputs, mask=mask)

    def infer_output_shape(self, input_shape):
        """Infers output shape from input using the specified reduction mode.

        :param input_shape: The shape of the input, which is typically [batch x sequence length x embedding size].

        :param return: The output shape after reduction.
        """
        if self._reduce_mode in {None, "none", "None"}:
            return input_shape
        elif self._reduce_mode == "concat":
            if len(input_shape) > 2:
                return input_shape[:-2] + (input_shape[-1] * input_shape[-2],)
            return input_shape
        else:
            return input_shape[:1] + input_shape[2:]  # Reduce sequence dimension (axis 1).


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
