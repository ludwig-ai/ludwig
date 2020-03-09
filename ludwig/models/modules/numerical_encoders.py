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

logger = logging.getLogger(__name__)


class NumericalPassthroughEncoder:

    def __init__(
            self,
            **kwargs
    ):
        pass

    def __call__(
            self,
            inputs,
            regularizer
    ):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.float32
            :type input_sequence: Tensor
        """
        return inputs

    def get_last_dimension(self):
        return 1
