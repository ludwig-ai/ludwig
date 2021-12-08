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
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from ludwig.utils.misc_utils import get_from_registry

optimizers_registry = {
    "sgd": torch.optim.SGD,
    "stochastic_gradient_descent": torch.optim.SGD,
    "gd": torch.optim.SGD,
    "gradient_descent": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
    #'ftrl': tf.keras.optimizers.Ftrl,
    #'nadam': tf.keras.optimizers.Nadam,
    "rmsprop": torch.optim.RMSprop,
}


@dataclass
class Clipper:
    clipglobalnorm: Optional[float] = 0.5
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None

    def clip_grads(self, variables: Iterable[torch.Tensor]):
        if self.clipglobalnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipvalue:
            torch.nn.utils.clip_grad_value_(variables, self.clipvalue)


def create_optimizer_with_clipper(
    model, type="sgd", clipglobalnorm=5.0, clipnorm=None, clipvalue=None, horovod=None, **kwargs
):
    optimizer_cls = get_from_registry(type.lower(), optimizers_registry)
    optimizer = create_optimizer(optimizer_cls, model, horovod, **kwargs)
    clipper = Clipper(clipglobalnorm=clipglobalnorm, clipnorm=clipnorm, clipvalue=clipvalue)
    return optimizer, clipper


def create_optimizer(optimizer_cls, model, horovod=None, **kwargs):
    optimizer = optimizer_cls(params=model.parameters(), **kwargs)
    if horovod:
        optimizer = horovod.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
        )
    return optimizer
