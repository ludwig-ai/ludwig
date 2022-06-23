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

from ludwig.constants import TYPE
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import initializer_registry


def _create_and_init(init_fn, init_kwargs, *args, **kwargs):
    t = torch.empty(*args, **kwargs)
    init_fn(t, **init_kwargs)
    return t


def get_initializer(parameters):
    if parameters is None:
        return lambda *args, **kwargs: _create_and_init(initializer_registry[parameters], {}, *args, **kwargs)
    elif isinstance(parameters, str):
        initializer_fun = get_from_registry(parameters, initializer_registry)
        return lambda *args, **kwargs: _create_and_init(initializer_fun, {}, *args, **kwargs)
    elif isinstance(parameters, dict):
        initializer_fun = get_from_registry(parameters[TYPE], initializer_registry)
        init_kwargs = parameters.copy()
        del init_kwargs[TYPE]
        return lambda *args, **kwargs: _create_and_init(initializer_fun, init_kwargs, *args, **kwargs)
    else:
        raise ValueError(
            f"Initializers parameters should be either strings or dictionaries, "
            f"but the provided parameters are a {type(parameters)}. "
            f"Parameters values: {parameters}"
        )
