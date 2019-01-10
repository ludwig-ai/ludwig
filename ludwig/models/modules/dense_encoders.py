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
from ludwig.models.modules.fully_connected_modules import FCStack


class Dense:
    def __init__(
            self,
            fc_layers=None,
            num_fc_layers=1,
            fc_size=256,
            norm=None,
            activation='relu',
            dropout=False,
            regularize=True,
            initializer=None,
            **kwargs
    ):
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_activation=activation,
            default_norm=norm,
            default_dropout=dropout,
            default_regularize=regularize,
            default_initializer=initializer
        )

    def __call__(
            self,
            input,
            input_size,
            regularizer,
            dropout_rate,
            is_training
    ):
        hidden = self.fc_stack(
            input,
            input_size,
            regularizer,
            dropout_rate,
            is_training=is_training
        )
        hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size
