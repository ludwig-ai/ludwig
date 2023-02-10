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

import torch

from ludwig.schema.initializers import InitializerConfig


def _create_and_init(init_fn, init_kwargs, *args, **kwargs):
    t = torch.empty(*args, **kwargs)
    init_fn(t, **init_kwargs)
    return t


def get_initializer(config: InitializerConfig):
    return lambda *args, **kwargs: _create_and_init(config.initializer_fn, config.initializer_params(), *args, **kwargs)
