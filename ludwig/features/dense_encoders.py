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


class DenseEncoder:

    def __init__(
            self,
            layers=None,
            num_layers=1,
            fc_size=256,
            activation='relu',
            use_bias=True,
            norm=None,
            dropout_rate=0,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            # activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
    ):
        self.fc_stack = FCStack(
            layers=layers,
            num_layers=num_layers,
            default_fc_size=fc_size,
            default_activation=activation,
            default_use_bias=use_bias,
            default_norm=norm,
            default_dropout_rate=dropout_rate,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            # default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
        )

    def __call__(self, inputs, training=None):
        return self.fc_stack(inputs, training=training)

    def get_last_dimension(self):
        if self.fc_stack.layers:
            return self.fc_stack.layers[-1]['fc_size']
        else:
            return None
