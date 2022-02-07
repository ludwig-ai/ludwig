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
from marshmallow_dataclass import dataclass

from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.registry import Registry
from ludwig.utils.schema_utils import (
    BaseMarshmallowConfig,
    create_cond,
    FloatRange,
    FloatRangeTupleDataclassField,
    get_custom_schema_from_marshmallow_class,
    NonNegativeFloat,
    StringOptions,
)

optimizer_registry = Registry()


def register_optimizer(name: str):
    def wrap(cls: BaseOptimizer):
        optimizer_registry[name] = (cls.torch_type, cls)
        return cls

    return wrap


@dataclass
class BaseOptimizer(BaseMarshmallowConfig):
    """Base class for optimizers. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding `torch.optim.Optimizer` class are copied over: check each class to check which attributes are
    different from the torch-specified defaults.
    """

    torch_type: ClassVar[Optional[torch.optim.Optimizer]] = None
    "Class variable pointing to the corresponding `torch.optim.Optimizer` class."
    type: str
    """Name corresponding to an optimizer `ludwig.modules.optimization_modules.optimizer_registry`.
       Technically mutable, but attempting to load a derived optimizer with `type` set to a mismatched value will
       result in a `ValidationError`."""

    @validates("type")
    def validate_type(self, data, **kwargs):
        """Workaround to enforce immutable `type` in defined optimizer classes.

        :param data: Any-typed object that should be a string correctly identifying the optimizer type.
        """
        if not isinstance(data, str):
            raise ValidationError(
                f"{self.__class__.__name__} expects type of field `type` to be `str`, instead received '{data}'"
            )
        default = self.declared_fields["type"].dump_default
        if default is not missing and data != default:
            # Handle aliases:
            if optimizer_registry[default] is optimizer_registry[data]:
                return
            raise ValidationError(
                f"{self.__class__.__name__} expects value of field `type` to be '{default}', instead received '{data}'"
            )


@register_optimizer(name="sgd")
@register_optimizer(name="gd")
@register_optimizer(name="stochastic_gradient_descent")
@register_optimizer(name="gradient_descent")
@dataclass
class SGDOptimizer(BaseOptimizer):
    """Parameters for stochastic gradient descent."""

    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    "Points to `torch.optim.SGD`."

    type: str = StringOptions(
        ["sgd", "gd", "stochastic_gradient_descent", "gradient_descent"], default="sgd", nullable=False
    )
    """Must be one of ['sgd', 'gd', 'stochastic_gradient_descent', 'gradient_descent']  - corresponds to names
       in `ludwig.modules.optimization_modules.optimizer_registry` (default: 'sgd')"""

    lr: float = FloatRange(default=0.001, min=0.0, max=1.0)
    "(default: 0.001)"

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = NonNegativeFloat(default=0)
    weight_decay: float = NonNegativeFloat(default=0)
    dampening: float = NonNegativeFloat(default=0)
    nesterov: bool = False


@register_optimizer(name="adam")
@dataclass
class AdamOptimizer(BaseOptimizer):
    """Parameters for adam optimization."""

    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    "Points to `torch.optim.Adam`."

    type: str = "adam"

    lr: float = FloatRange(default=0.001, min=0.0, max=1.0)
    "(default: 0.001)"

    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    "(default: (0.9, 0.999))"

    eps: float = NonNegativeFloat(default=1e-08)
    "(default: 1e-08)"

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    weight_decay: float = NonNegativeFloat(default=0)
    amsgrad: bool = False


@register_optimizer(name="adadelta")
@dataclass
class AdadeltaOptimizer(BaseOptimizer):
    """Parameters for adadelta optimization."""

    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    "Points to `torch.optim.Adadelta`."

    type: str = "adadelta"
    """Must be 'adadelta' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adadelta')"""

    rho: float = FloatRange(default=0.95, min=0.0, max=1.0)
    "(default: 0.95)"

    eps: float = NonNegativeFloat(default=1e-08)
    "(default: 1e-08)"

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    lr: float = FloatRange(default=1.0, min=0.0, max=1.0)
    weight_decay: float = NonNegativeFloat(default=0)


@register_optimizer(name="adagrad")
@dataclass
class AdagradOptimizer(BaseOptimizer):
    """Parameters for adagrad optimization."""

    # Example docstring
    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    "Points to `torch.optim.Adagrad`."

    type: str = "adagrad"
    """Must be 'adagrad' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adagrad')"""

    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    "(default: 0.1)"

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0)
    lr_decay: float = 0
    weight_decay: float = 0
    eps: float = 1e-10


@register_optimizer(name="adamax")
@dataclass
class AdamaxOptimizer(BaseOptimizer):
    """Parameters for adamax optimization."""

    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    "Points to `torch.optim.Adamax`."

    type: str = "adamax"
    """Must be 'adamax' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamax')"""

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
    """Parameters for rmsprop optimization."""

    torch_type: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    "Points to `torch.optim.RMSprop`."

    type: str = "rmsprop"
    """Must be 'rmsprop' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'rmsprop')"""

    weight_decay: float = NonNegativeFloat(default=0.9)
    "(default: 0.9)"

    momentum: float = NonNegativeFloat(default=0.0)
    "(default: 0.0)"

    eps: float = NonNegativeFloat(default=1e-10)
    "(default: 1e-10)"

    centered: bool = False
    "(default: False)"

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0)
    alpha: float = NonNegativeFloat(default=0.99)


def get_optimizer_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for optimizer in optimizer_registry:
        optimizer_cls = optimizer_registry[optimizer][1]
        preproc_cond = create_cond(
            {"type": optimizer},
            get_custom_schema_from_marshmallow_class(optimizer_cls),
        )
        conds.append(preproc_cond)
    return conds


class OptimizerMarshmallowField(fields.Field):
    """Custom marshmallow field that deserializes a dict to a valid optimizer from
    `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema for
    external usage."""

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
        return {"anyOf": [{"type": "null"}, *list(get_all_optimizer_json_schemas().values())]}


def OptimizerDataclassField(default={"type": "adam"}):
    """Custom dataclass field that when used inside of a dataclass will allow any optimizer in
    `ludwig.modules.optimization_modules.optimizer_registry`.

    Does not allow `None`, sets default optimizer to 'adam'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load optimizer from registry with given params. (default: {"type": "adam"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """
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
    """Return a dict of strings, wherein each key is an optimizer name pointing to its stringified JSON schema."""
    optimizer_schemas_json = {}
    for opt in optimizer_registry:
        schema_cls = optimizer_registry[opt][1]
        optimizer_schemas_json[opt] = get_custom_schema_from_marshmallow_class(schema_cls)
    return optimizer_schemas_json


@dataclass
class Clipper(BaseMarshmallowConfig):
    """Dataclass that holds gradient clipping parameters."""

    clipglobalnorm: Optional[float] = 0.5
    "(default: 0.5)"
    clipnorm: Optional[float] = None
    "(default: None)"
    clipvalue: Optional[float] = None
    "(default: None)"

    def clip_grads(self, variables: Iterable[torch.Tensor]):
        if self.clipglobalnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipvalue:
            torch.nn.utils.clip_grad_value_(variables, self.clipvalue)


def create_optimizer_with_clipper(
    model, optimizer: BaseOptimizer = SGDOptimizer(), clipper: Clipper = Clipper(clipglobalnorm=5.0), horovod=None
):
    """Returns a tuple of a ready-to-use torch optimizer and gradient clipping options.

    Creates an appropriately distributed torch optimizer, returns it and a `ludwig.modules.optimization_modules.Clipper`
    object for gradient clipping.

    :param model: Underlying Ludwig model
    :param optimizer: Instance of `ludwig.modules.optimization_modules.BaseOptimizer` (default:
           `ludwig.modules.optimization_modules.SGDOptimizer()`).
    :param clipper: Instance of `ludwig.modules.optimization_modules.Clipper` (default:
           `ludwig.modules.optimization_modules.Clipper(clipglobalnorm=5.0)`)
    """
    optimizer_cls = get_from_registry(optimizer.type.lower(), {k: v[0] for k, v in optimizer_registry.items()})
    cls_kwargs = {k: optimizer.__dict__[k] for k in optimizer.__dict__ if k != "type"}
    torch_optimizer: torch.optim.Optimizer = create_optimizer(optimizer_cls, model, horovod, **cls_kwargs)
    return torch_optimizer, clipper


def create_optimizer(optimizer_cls: BaseOptimizer, model, horovod=None, **kwargs) -> torch.optim.Optimizer:
    """Returns a ready-to-use torch optimizer instance based on the given optimizer.

    Takes a given `torch.optim.Optimizer` class and set of attributes via `kwargs` and constructs and returns a
    ready-to-use optimizer for training.

    :param optimizer_cls: Reference to one of the optimizer classes defined in
        `ludwig.modules.optimization_modules.optimizer_registry`.
    :param model: Instance of `ludwig.api.LudwigModel.
    :param horovod: Horovod parameters (default: None).
    :return: Initialized instance of a torch optimizer.
    """
    optimizer = optimizer_cls(params=model.parameters(), **kwargs)
    if horovod:
        optimizer = horovod.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
        )
    return optimizer


class ClipperMarshmallowField(fields.Field):
    """Custom marshmallow field class for gradient clipping.

    Deserializes a dict to a valid instance of `ludwig.modules.optimization_modules.Clipper` and creates a corresponding
    JSON schema for external usage.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, dict):
            try:
                return Clipper.Schema().load(value)
            except (TypeError, ValidationError):
                raise ValidationError(f"Invalid params for clipper: {value}, see Clipper class.")
        raise ValidationError("Field should be dict")

    def _jsonschema_type_mapping(self):
        return {"oneOf": [{"type": "null"}, get_custom_schema_from_marshmallow_class(Clipper)]}


def ClipperDataclassField(default={}, allow_none=True):
    """Returns custom dataclass field for `ludwig.modules.optimization_modules.Clipper`. Allows `None` by default.

    :param default: dict that specifies clipper param values that will be loaded by its schema class (default: {}).
    :param allow_none: Whether this field can accept `None` as a value. (default: True)
    """
    return field(
        metadata={"marshmallow_field": ClipperMarshmallowField(allow_none=allow_none)},
        default_factory=lambda: Clipper.Schema().load(default),
    )
