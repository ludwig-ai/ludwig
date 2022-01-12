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
from typing import ClassVar, Dict, Iterable, Optional, Tuple

import torch
from marshmallow import fields, missing, ValidationError
from marshmallow.decorators import validates
from marshmallow.utils import RAISE
from marshmallow_dataclass import dataclass
from marshmallow_jsonschema import JSONSchema as js

from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.registry import Registry
from ludwig.utils.schema_utils import FloatRange, FloatRangeTupleDataclassField, NonNegativeFloat, StringOptions

optimizer_registry = Registry()


def register_optimizer(name: str):
    def wrap(cls: BaseOptimizer):
        optimizer_registry[name] = (cls.torch_type, cls)
        return cls

    return wrap


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
    optimizer_cls = get_from_registry(type.lower(), {k: v[0] for k, v in optimizer_registry.items()})
    clipper = kwargs["clipper"]
    if clipper is None:
        clipper = Clipper(clipglobalnorm=clipglobalnorm, clipnorm=clipnorm, clipvalue=clipvalue)
    kwargs.pop("clipper", None)
    optimizer = create_optimizer(optimizer_cls, model, horovod, **kwargs)
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
                return Clipper.Schema().load(value)
            except (TypeError, ValidationError):
                raise ValidationError(f"Invalid params for clipper: {value}, see Clipper class.")
        raise ValidationError("Field should be dict")

    def _jsonschema_type_mapping(self):
        return {"oneOf": [{"type": "null"}, js().dump(Clipper.Schema())["definitions"]["Clipper"]]}


def ClipperDataclassField(default={}):
    return field(
        metadata={"marshmallow_field": ClipperMarshmallowField(allow_none=False)},
        default_factory=lambda: Clipper.Schema().load(default),
    )


@dataclass
class BaseOptimizer:
    torch_type: ClassVar[Optional[torch.optim.Optimizer]] = None
    type: str  # StringOptions(optimizer_registry.keys(), default=None)

    # Workaround to enforce immutable `type` in defined optimizer classes:
    @validates("type")
    def validate_type(self, data, **kwargs):
        default = self.declared_fields["type"].dump_default
        if default is not missing and data != default:
            raise ValidationError(
                f"{self.__class__.__name__} expects field `type` to be '{default}', instead received '{data}'"
            )

    class Meta:
        unknown = RAISE


@register_optimizer(name="sgd")
@register_optimizer(name="gd")
@register_optimizer(name="stochastic_gradient_descent")
@register_optimizer(name="gradient_descent")
@dataclass
class SGDOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    type: str = StringOptions(
        ["sgd", "gd", "stochastic_gradient_descent", "gradient_descent"], default="sgd", nullable=False
    )
    lr: float = FloatRange(default=0.001, min=0.0, max=1.0)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = NonNegativeFloat(default=0)
    weight_decay: float = NonNegativeFloat(default=0)
    dampening: float = NonNegativeFloat(default=0)
    nesterov: bool = False


# TODO: Check range limits/validation in the below classes? Also some hyperparameters supported by Torch currently do
# not have defaults in defaults.py (e.g. "lr" for all the below), so I haven't added them here yet.
@register_optimizer(name="adam")
@dataclass
class AdamOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    type: str = "adam"
    lr: float = FloatRange(default=0.001, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    weight_decay: float = NonNegativeFloat(default=0)
    amsgrad: bool = False


@register_optimizer(name="adadelta")
@dataclass
class AdadeltaOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    type: str = "adadelta"
    rho: float = FloatRange(default=0.95, min=0.0, max=1.0)
    eps: float = NonNegativeFloat(default=1e-08)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    lr: float = FloatRange(default=1.0, min=0.0, max=1.0)
    weight_decay: float = NonNegativeFloat(default=0)


@register_optimizer(name="adagrad")
@dataclass
class AdagradOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    type: str = "adagrad"
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0)
    lr_decay: float = 0
    weight_decay: float = 0
    eps: float = 1e-10


@register_optimizer(name="adamax")
@dataclass
class AdamaxOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    type: str = "adamax"
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0)


# NOTE: keep ftrl and nadam optimizers out of registry:
# @register_optimizer(name="ftrl")
@dataclass
class FtrlOptimizer(BaseOptimizer):
    # torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Ftrl
    type: str = "ftrl"
    learning_rate_power: float = FloatRange(default=-0.5, max=0.0)
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    l1_regularization_strength: float = NonNegativeFloat(default=0.0)
    l2_regularization_strength: float = NonNegativeFloat(default=0.0)


# @register_optimizer(name="nadam")
@dataclass
class NadamOptimizer(BaseOptimizer):
    # torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Nadam
    type: str = "nadam"
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0)
    momentum_decay: float = NonNegativeFloat(default=4e-3)


@register_optimizer(name="rmsprop")
@dataclass
class RMSPropOptimizer(BaseOptimizer):
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    type: str = "rmsprop"
    weight_decay: float = NonNegativeFloat(default=0.9)
    momentum: float = NonNegativeFloat(default=0.0)
    eps: float = NonNegativeFloat(default=1e-10)
    centered: bool = False
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    alpha: float = NonNegativeFloat(default=0.99)


class OptimizerMarshmallowField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dict):
            if "type" in value:
                opt = optimizer_registry[value["type"].lower()][1]
                try:
                    return opt.Schema().load(value)
                except (TypeError, ValidationError) as e:
                    raise ValidationError(f"Invalid params for optimizer: {value}, see `{opt}` definition. Error: {e}")
            raise ValidationError(
                f"Invalid params for optimizer: {value}, expect dict with at least a `type` attribute."
            )
        raise ValidationError("Field should be dict")

    def _jsonschema_type_mapping(self):
        return {"oneOf": [{"type": "null"}, *list(get_all_optimizer_json_schemas().values())]}


def OptimizerDataclassField(default={"type": "adam"}):
    try:
        opt = optimizer_registry[default["type"].lower()][1]
        if default["type"] not in optimizer_registry:
            raise ValidationError
        return field(
            metadata={"marshmallow_field": OptimizerMarshmallowField(allow_none=False)},
            default_factory=lambda: opt.Schema().load(default),
        )
    except (TypeError, ValidationError) as e:
        raise ValidationError(f"Unsupported optimizer type: {default['type']}. See optimizer_registry. Details: {e}")


def get_all_optimizer_json_schemas() -> Dict[str, str]:
    optimizer_schemas_json = {}
    for opt in optimizer_registry:
        schema_cls = optimizer_registry[opt][1]
        optimizer_schemas_json[opt] = js().dump(schema_cls.Schema())["definitions"][schema_cls.__name__]
    return optimizer_schemas_json
