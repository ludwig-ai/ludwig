# Copyright (c) 2022 Predibase, Inc.
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
from abc import abstractmethod
from functools import lru_cache
from typing import Dict

import numpy as np
import torch
from marshmallow_dataclass import dataclass
from torch.nn import Module, ModuleDict

from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.registry import Registry

module_registry = Registry()


def register_module(cls, name: str = None):
    """Registers a serializable module with a specified name.

    If name not specified, default to the name of the class
    """
    if name is None:
        name = cls.__name__
    module_registry[name] = cls
    return cls


@dataclass
class LudwigModuleState:
    """The state of a ludwig module, can be used to serialize or restore a saved madule."""

    type: str  # Module Type
    ludwig_version: str  # Version of ludwig which saved this object
    config: Dict  # Module Config
    metadata: Dict  # Preprocessing data (same a straining_set_metadata.json)
    saved_weights: Dict[str, np.ndarray]  # Saved weights of this module
    children: Dict[str, "LudwigModuleState"]  # Child modules


class LudwigModule(Module):
    """Base class for Ludwig modules which are implemented using PyTorch (inheriting from torch.nn.Module).

    Subclasses must implement @property input_shape() and forward().

    Provides a mechanism for adding custom loss terms by calling update_loss().

    To support serialization, subclasses must implement get_state() and @classmethod restore_from_state().
    """

    def __init__(self):
        super().__init__()
        self._losses = {}
        self.register_buffer("device_tensor", torch.zeros(0))

    @property
    def device(self):
        return self.device_tensor.device

    def get_state(self, config=None, metadata=None, saved_weights=None, children=None) -> LudwigModuleState:
        """Partial implementation of get_state which provides serialization support for torch weights.

        Subclasses should override get_state and provide their config, metadata, and child module state.
        """
        if saved_weights is None:
            saved_weights = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        return LudwigModuleState(
            type=type(self).__name__,
            ludwig_version=LUDWIG_VERSION,
            config={} if config is None else config,
            metadata={} if metadata is None else metadata,
            saved_weights={} if saved_weights is None else saved_weights,
            children={} if children is None else children,
        )

    def losses(self):
        collected_losses = []
        for loss in self._losses.values():
            collected_losses.append(loss)

        for child in self.children():
            if isinstance(child, LudwigModule):
                collected_losses.extend(child.losses())
            elif isinstance(child, ModuleDict):
                for c in child.values():
                    if hasattr(c, "losses"):  # Some modules, i.e. SequenceReducers, don't have losses.
                        collected_losses.extend(c.losses())
            elif isinstance(child, Module):
                pass
            else:
                raise ValueError

        return collected_losses

    def update_loss(self, key: str, loss: torch.Tensor):
        """This should be called in the forward pass to add a custom loss term to the combined loss.

        The loss tensor added here will be collected in the backward pass and added to the overall training loss to be
        minimized.
        """
        self._losses[key] = loss

    @property
    def input_dtype(self):
        return torch.float32

    @property
    @abstractmethod
    def input_shape(self) -> torch.Size:
        """Returns size of the input tensor without the batch dimension."""
        raise NotImplementedError("Abstract class.")

    @property
    def output_shape(self) -> torch.Size:
        """Returns size of the output tensor without the batch dimension."""
        return self._compute_output_shape()

    @lru_cache(maxsize=1)
    def _compute_output_shape(self) -> torch.Size:
        dummy_input = torch.rand(2, *self.input_shape, device=self.device)
        output_tensor = self.forward(dummy_input.type(self.input_dtype))

        if isinstance(output_tensor, torch.Tensor):
            return output_tensor.size()[1:]
        elif isinstance(output_tensor, dict) and "encoder_output" in output_tensor:
            return output_tensor["encoder_output"].size()[1:]
        else:
            raise ValueError("Unknown output tensor type.")
