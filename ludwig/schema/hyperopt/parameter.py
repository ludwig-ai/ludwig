from typing import List, Optional, Type, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.utils import register_parameter_config
from ludwig.schema.utils import ludwig_dataclass


def quantization_number_field(dtype: Union[Type[float], Type[int]] = float, default=None):
    description = (
        "Quantization number. Output values will be rounded to the nearest increment of `q` in range."
        "Quantization makes the upper bound inclusive."
    )
    if dtype is int:
        field = schema_utils.Integer(default=default, allow_none=True, description=description)
    else:
        field = schema_utils.Float(default=default, allow_none=True, description=description)

    return field


def log_base_field(default=10):
    return schema_utils.Float(default=default, description="Logarithmic base.")


@DeveloperAPI
@ludwig_dataclass
class BaseParameterConfig(schema_utils.BaseMarshmallowConfig):
    space: Optional[str] = None


@DeveloperAPI
@register_parameter_config("choice")
@ludwig_dataclass
class CategoricalParameterConfig(BaseParameterConfig):
    """Config for a randomly sampled categorical search space."""

    space: str = schema_utils.ProtectedString("choice")

    categories: List = schema_utils.List(
        description=(
            "The list of values to use in creating the categorical space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        )
    )


@DeveloperAPI
@register_parameter_config("grid_search")
@ludwig_dataclass
class GridSearchParameterConfig(BaseParameterConfig):
    """Config for a grid search space."""

    space: str = schema_utils.ProtectedString("grid_search")

    values: List = schema_utils.List(
        description=(
            "The list of values to use in creating the grid search space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        )
    )


@DeveloperAPI
@register_parameter_config("uniform")
@ludwig_dataclass
class UniformParameterConfig(BaseParameterConfig):
    """Config for a real-valued uniform search space."""

    space: str = schema_utils.ProtectedString("uniform")

    lower: float = schema_utils.Float(default=None, description="The minimum value the parameter can have.")

    upper: float = schema_utils.Float(default=None, description="The maximum value the parameter can have.")


@DeveloperAPI
@register_parameter_config("quniform")
@ludwig_dataclass
class QUniformParameterConfig(UniformParameterConfig):
    """Config for a real-valued uniform search space with quantization."""

    space: str = schema_utils.ProtectedString("quniform")

    q: float = quantization_number_field()


@DeveloperAPI
@register_parameter_config("loguniform")
@ludwig_dataclass
class LogUniformParameterConfig(UniformParameterConfig):
    """Config for a log-scaled real-valued uniform numeric search space."""

    space: str = schema_utils.ProtectedString("loguniform")

    base: float = log_base_field()


@DeveloperAPI
@register_parameter_config("qloguniform")
@ludwig_dataclass
class QLogUniformParameterConfig(UniformParameterConfig):
    """Config for a log-scaled real-valued uniform search space with quantization."""

    space: str = schema_utils.ProtectedString("qloguniform")

    q: float = quantization_number_field()

    base: float = log_base_field()


@DeveloperAPI
@register_parameter_config("randn")
@ludwig_dataclass
class RandnParameterConfig(BaseParameterConfig):
    """Config for a Gaussian search space."""

    space: str = schema_utils.ProtectedString("randn")


@DeveloperAPI
@register_parameter_config("qrandn")
@ludwig_dataclass
class QRandnParameterConfig(BaseParameterConfig):
    """Config for a Gaussian search space with quantization."""

    space: str = schema_utils.ProtectedString("qrandn")

    q: float = quantization_number_field()


@DeveloperAPI
@register_parameter_config("randint")
@ludwig_dataclass
class RandintParameterConfig(BaseParameterConfig):
    """Config for an integer-valued uniform search space."""

    space: str = schema_utils.ProtectedString("randint")

    lower: int = schema_utils.Integer(default=None, description="The minimum value the parameter can have.")

    upper: int = schema_utils.Integer(default=None, description="The maximum value the parameter can have.")


@DeveloperAPI
@register_parameter_config("qrandint")
@ludwig_dataclass
class QRandintParameterConfig(RandintParameterConfig):
    """Config for an integer-valued uniform search space with quantization."""

    space: str = schema_utils.ProtectedString("qrandint")

    q: int = quantization_number_field(dtype=int)


@DeveloperAPI
@register_parameter_config("lograndint")
@ludwig_dataclass
class LogRandintParameterConfig(RandintParameterConfig):
    """Config for an log-scaled integer-valued search space."""

    space: str = schema_utils.ProtectedString("lograndint")

    base: float = log_base_field()


@DeveloperAPI
@register_parameter_config("qlograndint")
@ludwig_dataclass
class QLogRandintParameterConfig(RandintParameterConfig):
    """Config for an log-scaled integer-valued search space with quantization."""

    space: str = schema_utils.ProtectedString("qlograndint")

    q: int = quantization_number_field(dtype=int)

    base: float = log_base_field()
