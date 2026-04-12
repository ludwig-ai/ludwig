# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
import dataclasses
from typing import Optional, TYPE_CHECKING

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
) -> tuple[type[torch.optim.Optimizer], dict]:
    """Returns the optimizer class and kwargs for the optimizer.

    :return: Tuple of optimizer class and kwargs for the optimizer.
    """
    from ludwig.schema.optimizers import optimizer_registry

    # Get the corresponding torch optimizer class for the given config:
    optimizer_cls = get_from_registry(optimizer_config.type.lower(), optimizer_registry)[0]

    # Create a dict of parameters to be passed to torch (i.e. everything except `type`):
    if dataclasses.is_dataclass(optimizer_config):
        config_dict = dataclasses.asdict(optimizer_config)
    elif hasattr(optimizer_config, "to_dict"):
        config_dict = optimizer_config.to_dict()
    else:
        config_dict = vars(optimizer_config)
    cls_kwargs = {field: value for field, value in config_dict.items() if field != "type"}

    # Most optimizers accept lr from Ludwig's trainer config. However, some optimizers
    # manage their own LR schedule and expect lr=None (e.g. Adafactor with relative_step=True).
    # Only override lr if the config does not already set it to None explicitly.
    if cls_kwargs.get("lr") is None and "lr" in cls_kwargs:
        # Config explicitly set lr=None (e.g. Adafactor relative_step mode) -- respect it.
        pass
    else:
        cls_kwargs["lr"] = learning_rate

    return optimizer_cls, cls_kwargs


def _get_loraplus_lr_ratio(model) -> float | None:
    """Check if the model has a LoRA+ lr ratio configured."""
    try:
        config_obj = getattr(model, "config_obj", None)
        if config_obj is None:
            return None
        adapter = getattr(config_obj, "adapter", None)
        if adapter is None:
            return None
        return getattr(adapter, "loraplus_lr_ratio", None)
    except Exception:
        return None


def _create_loraplus_param_groups(model, optimizer_kwargs, loraplus_lr_ratio):
    """Create separate param groups for LoRA A and B matrices with different learning rates.

    LoRA+ (Hayou et al., ICML 2024) uses a higher learning rate for B matrices and the base learning rate for A
    matrices. This provides 1-2% accuracy gain and up to 2x speedup.
    """
    import logging

    logger = logging.getLogger(__name__)

    base_lr = optimizer_kwargs["lr"]
    b_lr = base_lr * loraplus_lr_ratio

    a_params = []
    b_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            a_params.append(param)
        elif "lora_B" in name:
            b_params.append(param)
        else:
            other_params.append(param)

    logger.info(
        f"LoRA+ enabled: A matrices ({len(a_params)} params) lr={base_lr}, "
        f"B matrices ({len(b_params)} params) lr={b_lr}, "
        f"other ({len(other_params)} params) lr={base_lr}"
    )

    param_groups = []
    if a_params:
        param_groups.append({"params": a_params, "lr": base_lr})
    if b_params:
        param_groups.append({"params": b_params, "lr": b_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    return param_groups


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

    # LoRA+ support: use different learning rates for A and B matrices
    # (Hayou et al., ICML 2024). B matrices get lr * loraplus_lr_ratio.
    loraplus_lr_ratio = _get_loraplus_lr_ratio(model)
    if loraplus_lr_ratio is not None and loraplus_lr_ratio > 0:
        param_groups = _create_loraplus_param_groups(model, optimizer_kwargs, loraplus_lr_ratio)
        return optimizer_cls(param_groups, **{k: v for k, v in optimizer_kwargs.items() if k != "lr"})

    return optimizer_cls(model.parameters(), **optimizer_kwargs)
