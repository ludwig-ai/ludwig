"""Initializers come from PyTorch and are used to initialize the weights or bias of a model.

https://pytorch.org/docs/stable/nn.init.html
"""

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Dict, Union

import torch
from marshmallow import fields, ValidationError
from torch import nn

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

_initializer_registry = Registry()

_bias_initializer_registry = Registry()
"""A subset of the _initializer_registry, specifically for bias (1-dim) initializers."""


@DeveloperAPI
def register_initializer(name: str):
    """Register a weights initializer."""

    def wrap(initializer_config: InitializerConfig):
        _initializer_registry[name] = initializer_config
        return initializer_config

    return wrap


@DeveloperAPI
def register_bias_initializer(name: str):
    """Register a bias (1-dim) initializer."""

    def wrap(initializer_config: InitializerConfig):
        _bias_initializer_registry[name] = initializer_config
        return initializer_config

    return wrap


@DeveloperAPI
def get_initialize_cls(name: str):
    """Get the initializer schema class from the initializer schema class registry."""
    return _initializer_registry[name]


@DeveloperAPI
@ludwig_dataclass
class InitializerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for initializers.

    Not meant to be used directly.
    """

    type: str
    """Name corresponding to an initializer.

    Technically mutable, but attempting to load a derived initializer with `type` set to a mismatched value will result
    in a `ValidationError`.
    """

    def initializer_params(self) -> Dict[str, Any]:
        """Returns all params for this initializers without meta params."""
        params = self.to_dict()
        params.pop("type", None)
        return params

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call the initializer function on the given tensor."""
        raise NotImplementedError("Must implement initialization function manually in derived classes.")


@DeveloperAPI
@register_initializer(name="uniform")
@register_bias_initializer(name="uniform")
@ludwig_dataclass
class UniformInitializer(InitializerConfig):
    """Uniform initialization."""

    type: str = schema_utils.ProtectedString("uniform")

    a: float = schema_utils.FloatRange(default=0.5, description="The lower bound of the uniform distribution")

    b: float = schema_utils.FloatRange(default=1.0, description="The upper bound of the uniform distribution")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.uniform_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="normal")
@register_bias_initializer(name="normal")
@ludwig_dataclass
class NormalInitializer(InitializerConfig):
    """Normal initialization."""

    type: str = schema_utils.ProtectedString("normal")

    mean: float = schema_utils.FloatRange(default=0.0, description="The mean of the normal distribution")

    std: float = schema_utils.NonNegativeFloat(
        default=1.0, description="The standard deviation of the normal distribution"
    )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="trunc_normal")
@register_bias_initializer(name="trunc_normal")
@ludwig_dataclass
class TruncNormalInitializer(InitializerConfig):
    """Truncated normal initialization."""

    type: str = schema_utils.ProtectedString("trunc_normal")

    mean: float = schema_utils.FloatRange(default=0.0, description="The mean of the normal distribution")

    std: float = schema_utils.NonNegativeFloat(
        default=1.0, description="The standard deviation of the normal distribution"
    )

    a: float = schema_utils.FloatRange(default=-2.0, description="The minimum cutoff value.")

    b: float = schema_utils.FloatRange(default=2.0, description="The maximum cutoff value.")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.trunc_normal_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="constant")
@register_bias_initializer(name="constant")
@ludwig_dataclass
class ConstantInitializer(InitializerConfig):
    """Constant initialization."""

    type: str = schema_utils.ProtectedString("constant")

    val: float = schema_utils.FloatRange(default=0.0, description="The value to fill the tensor with")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.constant_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="ones")
@register_bias_initializer(name="ones")
@ludwig_dataclass
class OnesInitializer(InitializerConfig):
    """Ones initialization."""

    type: str = schema_utils.ProtectedString("ones")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.ones_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="zeros")
@register_bias_initializer(name="zeros")
@ludwig_dataclass
class ZerosInitializer(InitializerConfig):
    """Zeros initialization."""

    type: str = schema_utils.ProtectedString("zeros")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.zeros_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="identity")
@register_initializer(name="eye")
@ludwig_dataclass
class EyeInitializer(InitializerConfig):
    """Eye (identity matrix) initialization."""

    type: str = schema_utils.ProtectedString("eye")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.eye_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="xavier_uniform")
@ludwig_dataclass
class XavierUniformInitializer(InitializerConfig):
    """Xavier Uniform initialization."""

    type: str = schema_utils.ProtectedString("xavier_uniform")

    gain: float = schema_utils.FloatRange(default=1.0, description="An optional scaling factor")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.xavier_uniform_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="xavier_normal")
@ludwig_dataclass
class XavierNormalInitializer(InitializerConfig):
    """Xavier Normal initialization."""

    type: str = schema_utils.ProtectedString("xavier_normal")

    gain: float = schema_utils.FloatRange(default=1.0, description="An optional scaling factor")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.xavier_normal_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="kaiming_uniform")
@ludwig_dataclass
class KaimingUniformInitializer(InitializerConfig):
    """Kaiming Uniform initialization."""

    type: str = schema_utils.ProtectedString("kaiming_uniform")

    a: float = schema_utils.FloatRange(
        default=0.0,
        description="The negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)",
    )

    mode: str = schema_utils.StringOptions(
        ["fan_in", "fan_out"],
        default="fan_in",
        allow_none=False,
        description=(
            "Choosing ``'fan_in'`` preserves the magnitude of the variance of the weights in the forward pass. "
            "Choosing ``'fan_out'`` preserves the magnitudes in the backwards pass."
        ),
    )

    nonlinearity: str = schema_utils.StringOptions(
        ["relu", "leaky_relu"],
        default="leaky_relu",
        allow_none=False,
        description="The non-linear function",
    )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.kaiming_uniform_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="kaiming_normal")
@ludwig_dataclass
class KaimingNormalInitializer(InitializerConfig):
    """Kaiming Normal initialization."""

    type: str = schema_utils.ProtectedString("kaiming_normal")

    a: float = schema_utils.FloatRange(
        default=0.0,
        description="The negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)",
    )

    mode: str = schema_utils.StringOptions(
        ["fan_in", "fan_out"],
        default="fan_in",
        allow_none=False,
        description=(
            "Choosing ``'fan_in'`` preserves the magnitude of the variance of the weights in the forward pass. "
            "Choosing ``'fan_out'`` preserves the magnitudes in the backwards pass."
        ),
    )

    nonlinearity: str = schema_utils.StringOptions(
        ["relu", "leaky_relu"],
        default="leaky_relu",
        allow_none=False,
        description="The non-linear function",
    )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.kaiming_normal_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="orthogonal")
@ludwig_dataclass
class OrthogonalInitializer(InitializerConfig):
    """Orthogonal initialization."""

    type: str = schema_utils.ProtectedString("orthogonal")

    gain: float = schema_utils.FloatRange(default=1.0, description="An optional scaling factor")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.orthogonal_(tensor, **self.initializer_params())


@DeveloperAPI
@register_initializer(name="sparse")
@ludwig_dataclass
class SparseInitializer(InitializerConfig):
    """Sparse initialization."""

    type: str = schema_utils.ProtectedString("sparse")

    sparsity: float = schema_utils.NonNegativeFloat(
        default=0.1, max=1.0, description="The fraction of elements in each column to be set to zero"
    )

    std: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="The standard deviation of the normal distribution used to generate the non-zero values",
    )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.sparse_(tensor, **self.initializer_params())


@DeveloperAPI
def WeightsInitializerDataclassField(
    default: Union[str, Dict] = "xavier_uniform", description: str = "", parameter_metadata=None
):
    return _InitializerDataclassField(
        default, description, initializer_registry=_initializer_registry, parameter_metadata=parameter_metadata
    )


@DeveloperAPI
def BiasInitializerDataclassField(
    default: Union[str, Dict] = "xavier_uniform", description: str = "", parameter_metadata=None
):
    return _InitializerDataclassField(
        default, description, initializer_registry=_bias_initializer_registry, parameter_metadata=parameter_metadata
    )


@DeveloperAPI
def _InitializerDataclassField(
    default: Union[str, Dict], description: str, initializer_registry: Registry, parameter_metadata=None
):
    """Custom dataclass field that when used inside of a dataclass will allow any initializer.

    Args:
        default: Str or Dict specifying an initializer with a `type` field and its associated parameters. Will
            attempt to load an initializer with given params.
        description: Description of the initializer.
        single_dim: bool, if True, will only allow initializers that support single dimensional tensors like for bias.
    Returns:
        Initialized dataclass field that converts untyped dicts with params to initializer dataclass instances.
    """

    class InitializerMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict to a valid initializer and creates a corresponding
        `oneOf` JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None

            if isinstance(value, str):
                # If user provided the value as a string, assume they were providing the type
                value = {"type": value}

            if isinstance(value, dict):
                init_type = value.get("type")
                init_type = init_type.lower() if init_type else init_type
                if init_type in initializer_registry:
                    initializer = initializer_registry[init_type]
                    try:
                        return initializer.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for initializer: {value}, see `{initializer}` definition"
                        ) from e
                raise ValidationError(
                    f"Invalid initializer type: '{init_type}', expected one of: {list(initializer_registry.keys())}."
                )
            raise ValidationError(f"Invalid initializer param {value}, expected `None` or `dict`")

        @staticmethod
        def _jsonschema_type_mapping():
            accepted_keys = list(initializer_registry.keys())
            return {
                # Note that this uses the same conditional pattern as optimizers:
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": accepted_keys,
                        "default": default["type"],
                        "description": "The type of initializer to use during the learning process.",
                    },
                },
                "title": "initializer_options",
                "allOf": get_initializer_conds(initializer_registry),
                "required": ["type"],
                "description": description,
            }

    if isinstance(default, str):
        # If user provided the default as a string, assume they were providing the type
        default = {"type": default}

    if not isinstance(default, dict) or "type" not in default or default["type"] not in initializer_registry:
        raise ValidationError(f"Invalid default: `{default}`")

    try:
        initializer = initializer_registry[default["type"].lower()]
        load_default = lambda: initializer.Schema().load(default)
        dump_default = initializer.Schema().dump(default)

        # TODO(travis): use parameter_metadata
        return field(
            metadata={
                "marshmallow_field": InitializerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                    metadata={"description": description},
                )
            },
            default_factory=load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported initializer type: {default['type']}. See initializer_registry. Details: {e}"
        )


@DeveloperAPI
def get_initializer_conds(initializer_registry: Registry):
    """Returns a JSON schema of conditionals to validate against initializer types."""
    conds = []
    for initializer_name, initializer_cls in initializer_registry.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(initializer_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": initializer_name},
            other_props,
        )
        conds.append(preproc_cond)
    return conds
