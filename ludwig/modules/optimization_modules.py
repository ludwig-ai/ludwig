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
from dataclasses import asdict
from typing import Optional

import torch

from ludwig.distributed.base import DistributedStrategy
from ludwig.schema.optimizers import BaseOptimizerConfig, GradientClippingConfig, optimizer_registry, SGDOptimizerConfig
from ludwig.utils.misc_utils import get_from_registry


def create_clipper(gradient_clipping_config: Optional[GradientClippingConfig]):
    """Utility function that will convert a None-type gradient clipping config to the correct form."""
    if isinstance(gradient_clipping_config, GradientClippingConfig):
        return gradient_clipping_config
    # Return default config if provided value is None:
    return GradientClippingConfig()


def create_optimizer(
    model,
    learning_rate,
    distributed: DistributedStrategy,
    optimizer_config: BaseOptimizerConfig = SGDOptimizerConfig(),
):
    """Returns a ready-to-use torch optimizer instance based on the given optimizer config.

    :param model: Underlying Ludwig model
    :param learning_rate: Initial learning rate for the optimizer
    :param optimizer_config: Instance of `ludwig.modules.optimization_modules.BaseOptimizerConfig` (default:
           `ludwig.modules.optimization_modules.SGDOptimizerConfig()`).
    :param horovod: Horovod parameters (default: None).
    :return: Initialized instance of a torch optimizer.
    """
    # Get the corresponding torch optimizer class for the given config:
    optimizer_cls = get_from_registry(optimizer_config.type.lower(), optimizer_registry)[0]

    # Create a dict of parameters to be passed to torch (i.e. everything except `type`):
    cls_kwargs = {field: value for field, value in asdict(optimizer_config).items() if field != "type"}
    cls_kwargs["lr"] = learning_rate

    # Instantiate the optimizer:
    torch_optimizer: torch.optim.Optimizer = optimizer_cls(params=model.parameters(), **cls_kwargs)
    torch_optimizer = distributed.wrap_optimizer(torch_optimizer, model)
    return torch_optimizer
