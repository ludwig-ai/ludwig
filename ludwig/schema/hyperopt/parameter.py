from typing import List, Type, Union

from marshmallow.fields import Field

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.utils import register_parameter_config
from ludwig.schema.utils import ludwig_dataclass


def quantization_number_field(dtype: Union[Type[float], Type[int]] = float, default=None) -> Field:
    description = (
        "Quantization number. Output values will be rounded to the nearest increment of `q` in range."
        "Quantization makes the upper bound inclusive."
    )
    if dtype is int:
        field = schema_utils.Integer(default=default, allow_none=True, description=description)
    else:
        field = schema_utils.FloatRange(default=default, allow_none=True, description=description)

    return field


def log_base_field(default: float = 10) -> Field:
    return schema_utils.FloatRange(default=default, description="Logarithmic base.")


@DeveloperAPI
@register_parameter_config("choice")
@ludwig_dataclass
class ChoiceParameterConfig(schema_utils.BaseMarshmallowConfig):
    """Config for a randomly sampled categorical search space."""

    space: str = schema_utils.ProtectedString("choice")

    categories: List = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        description=(
            "The list of values to use in creating the categorical space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        ),
        field_options=[
            schema_utils.List(list_type=float, allow_none=False, description="The list of floats to randomly sample."),
            schema_utils.List(list_type=int, allow_none=False, description="The list of integers to randomly sample."),
            schema_utils.List(list_type=str, allow_none=False, description="The list of strings to randomly sample."),
            schema_utils.List(
                list_type=list,
                inner_type=dict,
                allow_none=False,
                description="The list of lists of configs to randomly sample.",
            ),
            schema_utils.DictList(allow_none=False, description="A list of nested config parameters to sample."),
        ],
    )


@DeveloperAPI
@register_parameter_config("grid_search")
@ludwig_dataclass
class GridSearchParameterConfig(schema_utils.BaseMarshmallowConfig):
    """Config for a grid search space."""

    space: str = schema_utils.ProtectedString("grid_search")

    values: List = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        description=(
            "The list of values to use in creating the grid search space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        ),
        field_options=[
            schema_utils.List(list_type=float, allow_none=False, description="The list of floats to randomly sample."),
            schema_utils.List(list_type=int, allow_none=False, description="The list of integers to randomly sample."),
            schema_utils.List(list_type=str, allow_none=False, description="The list of strings to randomly sample."),
        ],
    )


@DeveloperAPI
@register_parameter_config("uniform")
@ludwig_dataclass
class UniformParameterConfig(schema_utils.BaseMarshmallowConfig):
    """Config for a real-valued uniform search space."""

    space: str = schema_utils.ProtectedString("uniform")

    lower: float = schema_utils.FloatRange(default=None, description="The minimum value the parameter can have.")

    upper: float = schema_utils.FloatRange(default=None, description="The maximum value the parameter can have.")


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
class RandnParameterConfig(schema_utils.BaseMarshmallowConfig):
    """Config for a Gaussian search space."""

    space: str = schema_utils.ProtectedString("randn")

    mean: float = schema_utils.FloatRange(default=0.0, description="Mean of the  normal distribution.")

    sd: float = schema_utils.FloatRange(default=1.0, description="Standard deviation of the normal distribution.")


@DeveloperAPI
@register_parameter_config("qrandn")
@ludwig_dataclass
class QRandnParameterConfig(RandnParameterConfig):
    """Config for a Gaussian search space with quantization."""

    space: str = schema_utils.ProtectedString("qrandn")

    q: float = quantization_number_field()


@DeveloperAPI
@register_parameter_config("randint")
@ludwig_dataclass
class RandintParameterConfig(schema_utils.BaseMarshmallowConfig):
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
