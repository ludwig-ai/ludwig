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
from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING

import torch

from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule

if TYPE_CHECKING:
    from ludwig.schema.optimizers import BaseOptimizerConfig, GradientClippingConfig


def create_clipper(gradient_clipping_config: Optional["GradientClippingConfig"]):
    from ludwig.schema.optimizers import GradientClippingConfig

    """Utility function that will convert a None-type gradient clipping config to the correct form."""
    if isinstance(gradient_clipping_config, GradientClippingConfig):
        return gradient_clipping_config
    # Return default config if provided value is None:
    return GradientClippingConfig()


def get_optimizer_class_and_kwargs(
    optimizer_config: "BaseOptimizerConfig", learning_rate: float
) -> Tuple[Type[torch.optim.Optimizer], Dict]:
    """Returns the optimizer class and kwargs for the optimizer.

    :return: Tuple of optimizer class and kwargs for the optimizer.
    """
    from ludwig.schema.optimizers import optimizer_registry

    # Get the corresponding torch optimizer class for the given config:
    optimizer_cls = get_from_registry(optimizer_config.type.lower(), optimizer_registry)[0]

    # Create a dict of parameters to be passed to torch (i.e. everything except `type`):
    cls_kwargs = {field: value for field, value in asdict(optimizer_config).items() if field != "type"}
    cls_kwargs["lr"] = learning_rate

    return optimizer_cls, cls_kwargs


def create_optimizer(
    model: LudwigModule,
    optimizer_config: "BaseOptimizerConfig",
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Returns a ready-to-use torch optimizer instance based on the given optimizer config.

    :param model: Underlying Ludwig model
    :param learning_rate: Initial learning rate for the optimizer
    :param optimizer_config: Instance of `ludwig.modules.optimization_modules.BaseOptimizerConfig`.
    :return: Initialized instance of a torch optimizer.
    """
    # Make sure the optimizer is compatible with the available resources:
    if (optimizer_config.is_paged or optimizer_config.is_8bit) and (
        not torch.cuda.is_available() or torch.cuda.device_count() == 0
    ):
        raise ValueError(
            "Cannot use a paged or 8-bit optimizer on a non-GPU machine. "
            "Please use a different optimizer or run on a machine with a GPU."
        )

    optimizer_cls, optimizer_kwargs = get_optimizer_class_and_kwargs(optimizer_config, learning_rate)
    return optimizer_cls(model.parameters(), **optimizer_kwargs)
