from abc import ABC
from dataclasses import field
from typing import ClassVar, Optional, Tuple

import torch
from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema.utils import (
    BaseMarshmallowConfig,
    Boolean,
    create_cond,
    FloatRange,
    FloatRangeTupleDataclassField,
    NonNegativeFloat,
    StringOptions,
    unload_jsonschema_from_marshmallow_class,
)
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

    type: str = StringOptions(["sgd"], default="sgd", allow_none=False)
    """Must be 'sgd' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'sgd')"""

    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0, description="Learning rate.")

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = NonNegativeFloat(default=0.0, description="Momentum factor.")
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")
    dampening: float = NonNegativeFloat(default=0.0, description="Dampening for momentum.")
    nesterov: bool = Boolean(default=False, description="Enables Nesterov momentum.")


@register_optimizer(name="adam")
@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adam optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    """Points to `torch.optim.Adam`."""

    type: str = StringOptions(["adam"], default="adam", allow_none=False)
    """Must be 'adam' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adam')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0, description="Learning rate.")
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(
        default=(0.9, 0.999), description="Coefficients used for computing running averages of gradient and its square."
    )
    eps: float = NonNegativeFloat(
        default=1e-08, description="Term added to the denominator to improve numerical stability."
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay (L2 penalty).")
    amsgrad: bool = Boolean(
        default=False,
        description=(
            "Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam and"
            "Beyond'."
        ),
    )


@register_optimizer(name="adamw")
@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamw optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.AdamW
    """Points to `torch.optim.AdamW`."""

    type: str = StringOptions(["adamw"], default="adamw", allow_none=False)
    """Must be 'adamw' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamw')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    lr: float = FloatRange(default=1e-03, min=0.0, max=1.0, description="Learning rate.")
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(
        default=(0.9, 0.999), description="Coefficients used for computing running averages of gradient and its square."
    )
    eps: float = NonNegativeFloat(
        default=1e-08, description="Term added to the denominator to improve numerical stability."
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")
    amsgrad: bool = Boolean(
        default=False,
        description=(
            "Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam and"
            "Beyond'."
        ),
    )


@register_optimizer(name="adadelta")
@dataclass
class AdadeltaOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adadelta optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    """Points to `torch.optim.Adadelta`."""

    type: str = StringOptions(["adadelta"], default="adadelta", allow_none=False)
    """Must be 'adadelta' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adadelta')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    rho: float = FloatRange(
        default=0.9,
        min=0.0,
        max=1.0,
        description="Coefficient used for computing a running average of squared gradients.",
    )
    eps: float = NonNegativeFloat(
        default=1e-06, description="Term added to the denominator to improve numerical stability."
    )
    lr: float = FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        description="Coefficient that scales delta before it is applied to the parameters.",
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")


@register_optimizer(name="adagrad")
@dataclass
class AdagradOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adagrad optimization."""

    # Example docstring
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    """Points to `torch.optim.Adagrad`."""

    type: str = StringOptions(["adagrad"], default="adagrad", allow_none=False)
    """Must be 'adagrad' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adagrad')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    initial_accumulator_value: float = NonNegativeFloat(default=0, description="")
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0, description="Learning rate.")
    lr_decay: float = FloatRange(default=0, description="Learning rate decay.")
    weight_decay: float = FloatRange(default=0, description="Weight decay ($L2$ penalty).")
    eps: float = FloatRange(default=1e-10, description="Term added to the denominator to improve numerical stability.")


@register_optimizer(name="adamax")
@dataclass
class AdamaxOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamax optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    """Points to `torch.optim.Adamax`."""

    type: str = StringOptions(["adamax"], default="adamax", allow_none=False)
    """Must be 'adamax' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamax')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0, description="Learning rate.")
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(
        default=(0.9, 0.999), description="Coefficients used for computing running averages of gradient and its square."
    )
    eps: float = NonNegativeFloat(
        default=1e-08, description="Term added to the denominator to improve numerical stability."
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")


# NOTE: keep ftrl and nadam optimizers out of registry:
# @register_optimizer(name="ftrl")
@dataclass
class FtrlOptimizerConfig(BaseOptimizerConfig):
    # optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Ftrl
    type: str = StringOptions(["ftrl"], default="ftrl", allow_none=False)
    learning_rate_power: float = FloatRange(default=-0.5, max=0.0)
    initial_accumulator_value: float = NonNegativeFloat(default=0.1)
    l1_regularization_strength: float = NonNegativeFloat(default=0.0)
    l2_regularization_strength: float = NonNegativeFloat(default=0.0)


# @register_optimizer(name="nadam")
@dataclass
class NadamOptimizerConfig(BaseOptimizerConfig):
    # optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Nadam
    type: str = StringOptions(["nadam"], default="nadam", allow_none=False)
    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam :
    lr: float = FloatRange(default=2e-3, min=0.0, max=1.0, description="Learning rate.")
    betas: Tuple[float, float] = FloatRangeTupleDataclassField(
        default=(0.9, 0.999), description="Coefficients used for computing running averages of gradient and its square."
    )
    eps: float = NonNegativeFloat(
        default=1e-08, description="Term added to the denominator to improve numerical stability."
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")
    momentum_decay: float = NonNegativeFloat(default=4e-3, description="Momentum decay.")


@register_optimizer(name="rmsprop")
@dataclass
class RMSPropOptimizerConfig(BaseOptimizerConfig):
    """Parameters for rmsprop optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    """Points to `torch.optim.RMSprop`."""

    type: str = StringOptions(["rmsprop"], default="rmsprop", allow_none=False)
    """Must be 'rmsprop' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'rmsprop')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    lr: float = FloatRange(default=1e-2, min=0.0, max=1.0, description="Learning rate.")
    momentum: float = NonNegativeFloat(default=0.0, description="Momentum factor.")
    alpha: float = NonNegativeFloat(default=0.99, description="Smoothing constant.")
    eps: float = NonNegativeFloat(
        default=1e-08, description="Term added to the denominator to improve numerical stability."
    )
    centered: bool = Boolean(
        default=False,
        description=(
            "If True, computes the centered RMSProp, and the gradient is normalized by an estimation of its variance."
        ),
    )
    weight_decay: float = NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")


def get_optimizer_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for optimizer in optimizer_registry:
        optimizer_cls = optimizer_registry[optimizer][1]
        other_props = unload_jsonschema_from_marshmallow_class(optimizer_cls)["properties"]
        other_props.pop("type")
        preproc_cond = create_cond(
            {"type": optimizer},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


def OptimizerDataclassField(default={"type": "adam"}, description="TODO"):
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
                    "type": {"type": "string", "enum": list(optimizer_registry.keys()), "default": default["type"]},
                },
                "title": "optimizer_options",
                "allOf": get_optimizer_conds(),
                "required": ["type"],
                "description": description,
            }

    if not isinstance(default, dict) or "type" not in default or default["type"] not in optimizer_registry:
        raise ValidationError(f"Invalid default: `{default}`")
    try:
        opt = optimizer_registry[default["type"].lower()][1]
        load_default = opt.Schema().load(default)
        dump_default = opt.Schema().dump(default)

        return field(
            metadata={
                "marshmallow_field": OptimizerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                    metadata={"description": description},
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported optimizer type: {default['type']}. See optimizer_registry. Details: {e}")


@dataclass
class GradientClippingConfig(BaseMarshmallowConfig):
    """Dataclass that holds gradient clipping parameters."""

    clipglobalnorm: Optional[float] = FloatRange(default=0.5, allow_none=True, description="")

    clipnorm: Optional[float] = FloatRange(default=None, allow_none=True, description="")

    clipvalue: Optional[float] = FloatRange(default=None, allow_none=True, description="")


def GradientClippingDataclassField(default={}, allow_none=True, description="TODO"):
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
            return {
                "oneOf": [
                    {"type": "null", "title": "disabled", "description": "Disable gradient clipping."},
                    {**unload_jsonschema_from_marshmallow_class(GradientClippingConfig), "title": "enabled_options"},
                ],
                "title": "gradient_clipping_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = GradientClippingConfig.Schema().load(default)
    dump_default = GradientClippingConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": GradientClippingMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description},
            )
        },
        default_factory=lambda: load_default,
    )
