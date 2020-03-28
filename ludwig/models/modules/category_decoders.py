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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer

logger = logging.getLogger(__name__)


class Regressor(Layer):

    def __init__(self, num_classes=None, **kwargs):
        super().__init__()
        self.dense = Dense(num_classes)  # todo add initialization etc.

    def call(self, inputs, **kwargs):
        return tf.squeeze(self.dense(inputs))
