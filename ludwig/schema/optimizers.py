from abc import ABC
from dataclasses import field
from typing import ClassVar, Dict, Optional, Tuple, Type

import torch
from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema.metadata import OPTIMIZER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

optimizer_registry = Registry()


@DeveloperAPI
def register_optimizer(name: str):
    def wrap(optimizer_config: BaseOptimizerConfig):
        optimizer_registry[name] = (optimizer_config.optimizer_class, optimizer_config)
        return optimizer_config

    return wrap


@DeveloperAPI
def get_optimizer_cls(name: str):
    """Get the optimizer schema class from the optimizer schema class registry."""
    return optimizer_registry[name][1]


@DeveloperAPI
@ludwig_dataclass
class BaseOptimizerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for optimizers. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding `torch.optim.Optimizer` class are copied over: check each class to check which attributes are
    different from the torch-specified defaults.
    """

    optimizer_class: ClassVar[Optional[torch.optim.Optimizer]] = None
    "Class variable pointing to the corresponding `torch.optim.Optimizer` class."

    type: str
    """Name corresponding to an optimizer `ludwig.modules.optimization_modules.optimizer_registry`.

    Technically mutable, but attempting to load a derived optimizer with `type` set to a mismatched value will result in
    a `ValidationError`.
    """


@DeveloperAPI
@register_optimizer(name="sgd")
@ludwig_dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    """Parameters for stochastic gradient descent."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.SGD
    """Points to `torch.optim.SGD`."""

    type: str = schema_utils.ProtectedString("sgd")
    """Must be 'sgd' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'sgd')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD :
    momentum: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Momentum factor.", parameter_metadata=OPTIMIZER_METADATA["momentum"]
    )
    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )
    dampening: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Dampening for momentum.", parameter_metadata=OPTIMIZER_METADATA["dampening"]
    )
    nesterov: bool = schema_utils.Boolean(
        default=False, description="Enables Nesterov momentum.", parameter_metadata=OPTIMIZER_METADATA["nesterov"]
    )


@DeveloperAPI
@register_optimizer(name="lbfgs")
@ludwig_dataclass
class LBFGSOptimizerConfig(BaseOptimizerConfig):
    """Parameters for stochastic gradient descent."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.LBFGS
    """Points to `torch.optim.LBFGS`."""

    type: str = schema_utils.ProtectedString("lbfgs")
    """Must be 'lbfgs' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry` (default:
       'lbfgs')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
    max_iter: int = schema_utils.Integer(
        default=20,
        description="Maximum number of iterations per optimization step.",
        parameter_metadata=OPTIMIZER_METADATA["max_iter"],
    )

    max_eval: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="Maximum number of function evaluations per optimization step. Default: `max_iter` * 1.25.",
        parameter_metadata=OPTIMIZER_METADATA["max_eval"],
    )

    tolerance_grad: float = schema_utils.NonNegativeFloat(
        default=1e-07,
        description="Termination tolerance on first order optimality.",
        parameter_metadata=OPTIMIZER_METADATA["tolerance_grad"],
    )

    tolerance_change: float = schema_utils.NonNegativeFloat(
        default=1e-09,
        description="Termination tolerance on function value/parameter changes.",
        parameter_metadata=OPTIMIZER_METADATA["tolerance_change"],
    )

    history_size: int = schema_utils.Integer(
        default=100, description="Update history size.", parameter_metadata=OPTIMIZER_METADATA["history_size"]
    )

    line_search_fn: str = schema_utils.StringOptions(
        ["strong_wolfe"],
        default=None,
        allow_none=True,
        description="Line search function to use.",
        parameter_metadata=OPTIMIZER_METADATA["line_search_fn"],
    )


@DeveloperAPI
@register_optimizer(name="adam")
@ludwig_dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adam optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adam
    """Points to `torch.optim.Adam`."""

    type: str = schema_utils.ProtectedString("adam")
    """Must be 'adam' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adam')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    betas: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay (L2 penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    amsgrad: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam "
        "and Beyond'.",
        parameter_metadata=OPTIMIZER_METADATA["amsgrad"],
    )


@DeveloperAPI
@register_optimizer(name="adamw")
@ludwig_dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamw optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.AdamW
    """Points to `torch.optim.AdamW`."""

    type: str = schema_utils.ProtectedString("adamw")
    """Must be 'adamw' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamw')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam :
    betas: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    amsgrad: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the AMSGrad variant of this algorithm from the paper 'On the Convergence of Adam "
        "and Beyond'. ",
        parameter_metadata=OPTIMIZER_METADATA["amsgrad"],
    )


@DeveloperAPI
@register_optimizer(name="adadelta")
@ludwig_dataclass
class AdadeltaOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adadelta optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adadelta
    """Points to `torch.optim.Adadelta`."""

    type: str = schema_utils.ProtectedString("adadelta")
    """Must be 'adadelta' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adadelta')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta :
    rho: float = schema_utils.FloatRange(
        default=0.9,
        min=0,
        max=1,
        description="Coefficient used for computing a running average of squared gradients.",
        parameter_metadata=OPTIMIZER_METADATA["rho"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )


@DeveloperAPI
@register_optimizer(name="adagrad")
@ludwig_dataclass
class AdagradOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adagrad optimization."""

    # Example docstring
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adagrad
    """Points to `torch.optim.Adagrad`."""

    type: str = schema_utils.ProtectedString("adagrad")
    """Must be 'adagrad' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adagrad')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad :
    initial_accumulator_value: float = schema_utils.NonNegativeFloat(
        default=0, description="", parameter_metadata=OPTIMIZER_METADATA["initial_accumulator_value"]
    )

    lr_decay: float = schema_utils.FloatRange(
        default=0, description="Learning rate decay.", parameter_metadata=OPTIMIZER_METADATA["lr_decay"]
    )

    weight_decay: float = schema_utils.FloatRange(
        default=0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    eps: float = schema_utils.FloatRange(
        default=1e-10,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )


@DeveloperAPI
@register_optimizer(name="adamax")
@ludwig_dataclass
class AdamaxOptimizerConfig(BaseOptimizerConfig):
    """Parameters for adamax optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Adamax
    """Points to `torch.optim.Adamax`."""

    type: str = schema_utils.ProtectedString("adamax")
    """Must be 'adamax' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'adamax')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax :
    betas: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )


# NOTE: keep ftrl and nadam optimizers out of registry:
# @register_optimizer(name="ftrl")
@DeveloperAPI
@ludwig_dataclass
class FtrlOptimizerConfig(BaseOptimizerConfig):
    # optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.Ftrl
    type: str = schema_utils.ProtectedString("ftrl")

    learning_rate_power: float = schema_utils.FloatRange(
        default=-0.5, max=0, parameter_metadata=OPTIMIZER_METADATA["learning_rate_power"]
    )

    initial_accumulator_value: float = schema_utils.NonNegativeFloat(
        default=0.1, parameter_metadata=OPTIMIZER_METADATA["initial_accumulator_value"]
    )

    l1_regularization_strength: float = schema_utils.NonNegativeFloat(
        default=0.0, parameter_metadata=OPTIMIZER_METADATA["l1_regularization_strength"]
    )

    l2_regularization_strength: float = schema_utils.NonNegativeFloat(
        default=0.0, parameter_metadata=OPTIMIZER_METADATA["l2_regularization_strength"]
    )


@DeveloperAPI
@register_optimizer(name="nadam")
@ludwig_dataclass
class NadamOptimizerConfig(BaseOptimizerConfig):
    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.NAdam
    """Points to `torch.optim.NAdam`."""

    type: str = schema_utils.ProtectedString("nadam")

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam :

    betas: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(0.9, 0.999),
        description="Coefficients used for computing running averages of gradient and its square.",
        parameter_metadata=OPTIMIZER_METADATA["betas"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Weight decay ($L2$ penalty).", parameter_metadata=OPTIMIZER_METADATA["weight_decay"]
    )

    momentum_decay: float = schema_utils.NonNegativeFloat(
        default=4e-3, description="Momentum decay.", parameter_metadata=OPTIMIZER_METADATA["momentum_decay"]
    )


@DeveloperAPI
@register_optimizer(name="rmsprop")
@ludwig_dataclass
class RMSPropOptimizerConfig(BaseOptimizerConfig):
    """Parameters for rmsprop optimization."""

    optimizer_class: ClassVar[torch.optim.Optimizer] = torch.optim.RMSprop
    """Points to `torch.optim.RMSprop`."""

    type: str = schema_utils.ProtectedString("rmsprop")
    """Must be 'rmsprop' - corresponds to name in `ludwig.modules.optimization_modules.optimizer_registry`
       (default: 'rmsprop')"""

    # Defaults taken from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop:
    momentum: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Momentum factor.",
        parameter_metadata=OPTIMIZER_METADATA["momentum"],
    )

    alpha: float = schema_utils.NonNegativeFloat(
        default=0.99,
        description="Smoothing constant.",
        parameter_metadata=OPTIMIZER_METADATA["alpha"],
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=1e-08,
        description="Term added to the denominator to improve numerical stability.",
        parameter_metadata=OPTIMIZER_METADATA["eps"],
    )

    centered: bool = schema_utils.Boolean(
        default=False,
        description="If True, computes the centered RMSProp, and the gradient is normalized by an estimation of its "
        "variance.",
        parameter_metadata=OPTIMIZER_METADATA["centered"],
    )

    weight_decay: float = schema_utils.NonNegativeFloat(default=0.0, description="Weight decay ($L2$ penalty).")


@DeveloperAPI
def get_optimizer_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for optimizer in optimizer_registry:
        optimizer_cls = optimizer_registry[optimizer][1]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(optimizer_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": optimizer},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def OptimizerDataclassField(default="adam", description="", parameter_metadata: ParameterMetadata = None):
    """Custom dataclass field that when used inside of a dataclass will allow any optimizer in
    `ludwig.modules.optimization_modules.optimizer_registry`.

    Sets default optimizer to 'adam'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load optimizer from registry with given params. (default: {"type": "adam"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """

    class OptimizerSelection(schema_utils.TypeSelection):
        """Custom marshmallow field that deserializes a dict to a valid optimizer from
        `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema
        for external usage."""

        def __init__(self):
            super().__init__(
                registry=optimizer_registry,
                default_value=default,
                description=description,
                parameter_metadata=parameter_metadata,
            )

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return get_optimizer_cls(key)

        @staticmethod
        def _jsonschema_type_mapping():
            # Note that this uses the same conditional pattern as combiners:
            return {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(optimizer_registry.keys()),
                        "default": default,
                        "description": "The type of optimizer to use during the learning process",
                    },
                },
                "title": "optimizer_options",
                "allOf": get_optimizer_conds(),
                "required": ["type"],
                "description": description,
            }

    return OptimizerSelection().get_default_field()


@DeveloperAPI
@ludwig_dataclass
class GradientClippingConfig(schema_utils.BaseMarshmallowConfig):
    """Dataclass that holds gradient clipping parameters."""

    clipglobalnorm: Optional[float] = schema_utils.FloatRange(
        default=0.5,
        allow_none=True,
        description="Maximum allowed norm of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )

    # TODO(travis): is this redundant with `clipglobalnorm`?
    clipnorm: Optional[float] = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        description="Maximum allowed norm of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )

    clipvalue: Optional[float] = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        description="Maximum allowed value of the gradients",
        parameter_metadata=OPTIMIZER_METADATA["gradient_clipping"],
    )


@DeveloperAPI
def GradientClippingDataclassField(description: str, default: Dict = {}):
    """Returns custom dataclass field for `ludwig.modules.optimization_modules.GradientClippingConfig`. Allows
    `None` by default.

    :param description: Description of the gradient dataclass field
    :param default: dict that specifies clipping param values that will be loaded by its schema class (default: {}).
    """
    allow_none = True

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

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "oneOf": [
                    {"type": "null", "title": "disabled", "description": "Disable gradient clipping."},
                    {
                        **schema_utils.unload_jsonschema_from_marshmallow_class(GradientClippingConfig),
                        "title": "enabled_options",
                    },
                ],
                "title": "gradient_clipping_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: GradientClippingConfig.Schema().load(default)
    dump_default = GradientClippingConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": GradientClippingMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(OPTIMIZER_METADATA["gradient_clipping"]),
                },
            )
        },
        default_factory=load_default,
    )
