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
from dataclasses import field
from typing import ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import torch
from marshmallow import fields, ValidationError
from marshmallow.decorators import validates
from marshmallow.utils import INCLUDE
from marshmallow_dataclass import dataclass
from marshmallow_jsonschema import JSONSchema as js

from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.schema_utils import FloatRange, NonNegativeFloat


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
    optimizer_cls = get_from_registry(type.lower(), {k: v[0] for k, v in get_optimizers_registry()})
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


class ClipperMarshmallowField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dict):
            try:
                return Clipper(**value)
            except (TypeError, ValidationError):
                raise ValidationError(f"Invalid params for clipper: {value}, see Clipper class.")
        raise ValidationError("Field should be dict")

    def _jsonschema_type_mapping(self):
        return {"oneOf": [{"type": None}, js().dump(Clipper.Schema())["definitions"]["Clipper"]]}


def ClipperDataclassField():
    return field(metadata={"marshmallow_field": ClipperMarshmallowField(allow_none=False)}, default_factory=Clipper)


@dataclass
class BaseOptimizer:
    torch_type: ClassVar[Optional[torch.optim.Optimizer]] = None
    type: Optional[str] = None
    clipper: Union[None, Clipper] = ClipperDataclassField()

    class Meta:
        unknown = INCLUDE


@dataclass
class SGDOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    type: str = "sgd"
    lr: float = FloatRange(default=0.001, min=0.0, max=1.0)


# TODO: Check range limits/validation in the below classes? Also some hyperparameters supported by Torch currently do
# not have defaults in defaults.py (e.g. "lr" for all the below), so I haven't added them here yet.
@dataclass
class AdamOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    type: str = "adam"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = NonNegativeFloat(default=1e-08)

    @validates("betas")
    def validateBetas(self, data):
        if isinstance(data, tuple) and list(map(type, data)) == [float, float]:
            if all(list(map(lambda b: 0.0 <= b <= 1.0, data))):
                return data
        raise ValidationError(f'Field "betas" should be of type "Tuple[float, float]", instead received: {data}')


@dataclass
class AdadeltaOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    type: str = "adadelta"
    rho: float = FloatRange(default=0.95, min=0.0, max=1.0)
    eps: float = NonNegativeFloat(default=1e-08)


@dataclass
class AdagradOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    type: str = "adagrad"
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)


# TODO: No vars for adamax?
@dataclass
class AdamaxOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    type: str = "adamax"


@dataclass
class FtrlOptimizer(BaseOptimizer):
    # torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Ftrl
    type: str = "ftrl"
    learning_rate_power: float = FloatRange(default=-0.5, max=0.0)
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    l1_regularization_strength: float = NonNegativeFloat(default=0.0)
    l2_regularization_strength: float = NonNegativeFloat(default=0.0)


# TODO: No vars for nadam?
@dataclass
class NadamOptimizer(BaseOptimizer):
    # torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Nadam
    type: str = "nadam"


@dataclass
class RMSPropOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    type: str = "rmsprop"
    weight_decay: float = NonNegativeFloat(default=0.9)
    momentum: float = NonNegativeFloat(default=0.0)
    eps: float = NonNegativeFloat(default=1e-10)
    centered: bool = False


class OptimizerMarshmallowField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        print("_deserialize")
        if isinstance(value, dict):
            lowercase = {k.lower() for k in value.keys()}
            if "type" in lowercase:
                try:
                    return get_optimizers_registry()["type"][1](**value)
                except (TypeError, ValidationError):
                    t = get_optimizers_registry()["type"][1]
                    raise ValidationError(f"Invalid params for optimizer: {value}, see `{t}` definition.")
            raise ValidationError(
                f"Invalid params for optimizer: {value}, expect dict with at least a `type` attribute."
            )
        raise ValidationError("Field should be dict")

    def _jsonschema_type_mapping(self):
        return {"oneOf": [{"type": None}, js().dump(BaseOptimizer.Schema())["definitions"]["BaseOptimizer"]]}


def OptimizerDataclassField(default={"type": "adam"}):
    return field(
        metadata={"marshmallow_field": OptimizerMarshmallowField(allow_none=False)},
        default_factory=lambda: BaseOptimizer.Schema().load(default),
    )


def get_optimizers_schemas() -> List[BaseOptimizer]:
    return BaseOptimizer.__subclasses__()


def get_optimizers_registry() -> Dict[str, Tuple[torch.optim.Optimizer, BaseOptimizer]]:
    schemas = {
        schema.type: (schema.torch_type, schema) for schema in get_optimizers_schemas() if schema.torch_type is not None
    }
    return {
        **schemas,
        **dict.fromkeys(
            ["sgd", "stochastic_gradient_descent", "gd", "gradient_descent"], (torch.optim.SGD, SGDOptimizer)
        ),
    }
