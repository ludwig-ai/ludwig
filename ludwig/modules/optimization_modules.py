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
from abc import ABC
from dataclasses import asdict, field
from typing import ClassVar, Dict, Iterable, Optional, Tuple

import torch
from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.marshmallow.marshmallow_schema_utils import (
    BaseMarshmallowConfig,
    create_cond,
    FloatRange,
    FloatRangeTupleDataclassField,
    get_custom_schema_from_marshmallow_class,
    NonNegativeFloat,
    StringOptions,
)
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.registry import Registry

optimizer_registry = Registry()


def register_optimizer(name: str):
    def wrap(optimizer_config: BaseOptimizerConfig):
        optimizer_registry[name] = (optimizer_config.optimizer_class, optimizer_config)
        return optimizer_config

    return wrap


@dataclass
class BaseOptimizerConfig(BaseMarshmallowConfig, ABC):
    """Base class for optimizers. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding `torch.optim.Optimizer` class are copied over: check each class to check which attributes are
    different from the torch-specified defaults.
    """

    optimizer_class: ClassVar[Optional[torch.optim.Optimizer]] = None
    "Class variable pointing to the corresponding `torch.optim.Optimizer` class."
    type: str
    """Name corresponding to an optimizer `ludwig.modules.optimization_modules.optimizer_registry`.
       Technically mutable, but attempting to load a derived optimizer with `type` set to a mismatched value will
       result in a `ValidationError`."""


@register_optimizer(name="sgd")
@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    """Parameters for stochastic gradient descent."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    """Points to `torch.optim.SGD`."""

    type: str = StringOptions(["sgd"], default="sgd", nullable=False)
    """Must be 'sgd' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'sgd')"""

    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0)
    """(default: 0.001)"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = NonNegativeFloat(default=0.0)
    weight_decay: float = NonNegativeFloat(default=0.0)
    dampening: float = NonNegativeFloat(default=0.0)
    nesterov: bool = False


@register_optimizer(name="adam")
@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adam optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    """Points to `torch.optim.Adam`."""

    type: str = StringOptions(["adam"], default="adam", nullable=False)
    """Must be 'adam' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adam')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0.0)
    amsgrad: bool = False


@register_optimizer(name="adamw")
@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamw optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.AdamW
    """Points to `torch.optim.AdamW`."""

    type: str = StringOptions(["adamw"], default="adamw", nullable=False)
    """Must be 'adamw' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamw')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0.0)
    amsgrad: bool = False


@register_optimizer(name="adadelta")
@dataclass
class AdadeltaOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adadelta optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    """Points to `torch.optim.Adadelta`."""

    type: str = StringOptions(["adadelta"], default="adadelta", nullable=False)
    """Must be 'adadelta' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adadelta')"""

    rho: float = FloatRange(default=0.95, min=0.0, max=1.0)
    """(default: 0.95)"""

    eps: float = NonNegativeFloat(default=1e-08)
    """(default: 1e-08)"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    lr: float = FloatRange(default=1.0, min=0.0, max=1.0)
    weight_decay: float = NonNegativeFloat(default=0.0)


@register_optimizer(name="adagrad")
@dataclass
class AdagradOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adagrad optimization."""

    # Example docstring
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    """Points to `torch.optim.Adagrad`."""

    type: str = StringOptions(["adagrad"], default="adagrad", nullable=False)
    """Must be 'adagrad' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adagrad')"""

    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    """(default: 0.1)"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0)
    lr_decay: float = 0
    weight_decay: float = 0
    eps: float = 1e-10


@register_optimizer(name="adamax")
@dataclass
class AdamaxOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamax optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    """Points to `torch.optim.Adamax`."""

    type: str = StringOptions(["adamax"], default="adamax", nullable=False)
    """Must be 'adamax' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamax')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0.0)


# NOTE: keep ftrl and nadam optimizers out of registry:
# @register_optimizer(name="ftrl")
@dataclass
class FtrlOptimizerConfig(BaseOptimizerConfig):
    # optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Ftrl
    type: str = StringOptions(["ftrl"], default="ftrl", nullable=False)
    learning_rate_power: float = FloatRange(default=-0.5, max=0.0)
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    l1_regularization_strength: float = NonNegativeFloat(default=0.0)
    l2_regularization_strength: float = NonNegativeFloat(default=0.0)


# @register_optimizer(name="nadam")
@dataclass
class NadamOptimizerConfig(BaseOptimizerConfig):
    # optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Nadam
    type: str = StringOptions(["nadam"], default="nadam", nullable=False)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0)
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(default=(0.9, 0.999))
    eps: float = NonNegativeFloat(default=1e-08)
    weight_decay: float = NonNegativeFloat(default=0.0)
    momentum_decay: float = NonNegativeFloat(default=4e-3)


@register_optimizer(name="rmsprop")
@dataclass
class RMSPropOptimizerConfig(BaseOptimizerConfig):
    """Parameters for rmsprop optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    """Points to `torch.optim.RMSprop`."""

    type: str = StringOptions(["rmsprop"], default="rmsprop", nullable=False)
    """Must be 'rmsprop' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'rmsprop')"""

    weight_decay: float = NonNegativeFloat(default=0.9)
    """(default: 0.9)"""

    momentum: float = NonNegativeFloat(default=0.0)
    """(default: 0.0)"""

    eps: float = NonNegativeFloat(default=1e-10)
    """(default: 1e-10)"""

    centered: bool = False
    """(default: False)"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0)
    alpha: float = NonNegativeFloat(default=0.99)


def get_optimizer_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for optimizer in optimizer_registry:
        optimizer_cls = optimizer_registry[optimizer][1]
        other_props = get_custom_schema_from_marshmallow_class(optimizer_cls)["properties"]
        other_props.pop("type")
        preproc_cond = create_cond(
            {"type": optimizer},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


def OptimizerDataclassField(default={"type": "adam"}):
    """Custom dataclass field that when used inside of a dataclass will allow any optimizer in
    `ludwig.modules.optimization_modules.optimizer_registry`.

    Sets default optimizer to 'adam'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load optimizer from registry with given params. (default: {"type": "adam"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """

    class OptimizerMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict to a valid optimizer from
        `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema
        for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if "type" in value and value["type"] in optimizer_registry:
                    opt = optimizer_registry[value["type"].lower()][1]
                    try:
                        return opt.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for optimizer: {value}, see `{opt}` definition. Error: {e}"
                        )
                raise ValidationError(
                    f"Invalid params for optimizer: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            # Note that this uses the same conditional pattern as combiners:
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(optimizer_registry.keys())},
                },
                #
                "allOf": get_optimizer_conds(),
                "required": ["type"],
            }

    if not isinstance(default, dict) or "type" not in default or default["type"] not in optimizer_registry:
        raise ValidationError(f"Invalid default: `{default}`")
    try:
        opt = optimizer_registry[default["type"].lower()][1]

        return field(
            metadata={"marshmallow_field": OptimizerMarshmallowField(allow_none=False)},
            default_factory=lambda: opt.Schema().load(default),
        )
    except Exception as e:
        raise ValidationError(f"Unsupported optimizer type: {default['type']}. See optimizer_registry. Details: {e}")


def get_all_optimizer_json_schemas() -> Dict[str, str]:
    """Return a dict of strings, wherein each key is an optimizer name pointing to its stringified JSON schema."""
    optimizer_schemas_json = {}
    for opt in optimizer_registry:
        schema_cls = optimizer_registry[opt][1]
        optimizer_schemas_json[opt] = get_custom_schema_from_marshmallow_class(schema_cls)
    return optimizer_schemas_json


@dataclass
class GradientClippingConfig(BaseMarshmallowConfig):
    """Dataclass that holds gradient clipping parameters."""

    clipglobalnorm: Optional[float] = 0.5
    """(default: 0.5)"""
    clipnorm: Optional[float] = None
    """(default: None)"""
    clipvalue: Optional[float] = None
    """(default: None)"""

    def clip_grads(self, variables: Iterable[torch.Tensor]):
        if self.clipglobalnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipnorm:
            torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
        if self.clipvalue:
            torch.nn.utils.clip_grad_value_(variables, self.clipvalue)


def GradientClippingDataclassField(default={}, allow_none=True):
    """Returns custom dataclass field for `ludwig.modules.optimization_modules.GradientClippingConfig`. Allows
    `None` by default.

    :param default: dict that specifies clipping param values that will be loaded by its schema class (default: {}).
    :param allow_none: Whether this field can accept `None` as a value. (default: True)
    """

    class GradientClippingMarshmallowField(fields.Field):
        """Custom marshmallow field class for gradient clipping.

        Deserializes a dict to a valid instance of `ludwig.modules.optimization_modules.GradientClippingConfig` and
        creates a corresponding JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return GradientClippingConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(
                        f"Invalid params for gradient clipping: {value}, see GradientClippingConfig class."
                    )
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {"oneOf": [{"type": "null"}, get_custom_schema_from_marshmallow_class(GradientClippingConfig)]}

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={"marshmallow_field": GradientClippingMarshmallowField(allow_none=allow_none)},
        default_factory=lambda: GradientClippingConfig.Schema().load(default),
    )


def create_clipper(gradient_clipping_config: Optional[GradientClippingConfig]):
    """Utility function that will convert a None-type gradient clipping config to the correct form."""
    if isinstance(gradient_clipping_config, GradientClippingConfig):
        return gradient_clipping_config
    # none_dict = dict.fromkeys(asdict(GradientClippingConfig()), None)
    # return GradientClippingConfig.Schema().load(none_dict)
    return GradientClippingConfig()


def create_optimizer(
    model,
    optimizer_config: BaseOptimizerConfig = SGDOptimizerConfig(),
    horovod=None,
):
    """Returns a ready-to-use torch optimizer instance based on the given optimizer config.

    :param model: Underlying Ludwig model
    :param optimizer_config: Instance of `ludwig.modules.optimization_modules.BaseOptimizerConfig` (default:
           `ludwig.modules.optimization_modules.SGDOptimizerConfig()`).
    :param horovod: Horovod parameters (default: None).
    :return: Initialized instance of a torch optimizer.
    """
    # Get the corresponding torch optimizer class for the given config:
    optimizer_cls = get_from_registry(optimizer_config.type.lower(), optimizer_registry)[0]

    # Create a dict of parameters to be passed to torch (i.e. everything except `type`):
    cls_kwargs = {field: value for field, value in asdict(optimizer_config).items() if field != "type"}

    # Instantiate the optimizer:
    torch_optimizer: torch.optim.Optimizer = optimizer_cls(params=model.parameters(), **cls_kwargs)
    if horovod:
        torch_optimizer = horovod.DistributedOptimizer(
            torch_optimizer,
            named_parameters=model.named_parameters(),
        )
    print(torch_optimizer)
    return torch_optimizer
