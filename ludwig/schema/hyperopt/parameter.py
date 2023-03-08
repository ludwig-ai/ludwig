from typing import Callable, List, Optional, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

parameter_type_registry = Registry()


def register_parameter_type(name: str) -> Callable[[Type["BaseParameterConfig"]], Type["BaseParameterConfig"]]:
    def wrap(cls: Type["BaseParameterConfig"]) -> Type["BaseParameterConfig"]:
        parameter_type_registry[name] = cls
        return cls

    return wrap


@DeveloperAPI
@ludwig_dataclass
class BaseParameterConfig(schema_utils.BaseMarshmallowConfig):
    space: Optional[str] = None


@DeveloperAPI
@register_parameter_type("choice")
@ludwig_dataclass
class CategoricalParameterConfig(BaseParameterConfig):
    space: str = schema_utils.ProtectedString("choice")

    categories: List = schema_utils.List(
        description=(
            "The list of values to use in creating the grid search space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        )
    )


@DeveloperAPI
@register_parameter_type("grid_search")
@ludwig_dataclass
class GridSearchParameterConfig(BaseParameterConfig):
    space: str = schema_utils.ProtectedString("grid_search")

    values: List = schema_utils.List(
        description=(
            "The list of values to use in creating the grid search space. The type of each value of the list is "
            "general, i.e., they could be strings, integers, floats and anything else, even entire dictionaries."
        )
    )


@DeveloperAPI
@register_parameter_type("uniform")
@register_parameter_type("quniform")
@register_parameter_type("loguniform")
@register_parameter_type("qloguniform")
@ludwig_dataclass
class UniformParameterConfig(BaseParameterConfig):
    space: str = schema_utils.ProtectedString("uniform")

    lower: float = schema_utils.Float(default=None, description="The minimum value the parameter can have.")

    upper: float = schema_utils.Float(default=None, description="The maximum value the parameter can have.")

    q: float = schema_utils.Float(
        default=1,
        description=("Quantization factor. Output values will be rounded to the nearest multiple of `q` in range."),
    )

    base: float = schema_utils.Float(default=None, allow_none=True, description="Logarithmic base. If supplied, will")


@DeveloperAPI
@register_parameter_type("randn")
@register_parameter_type("qrandn")
@ludwig_dataclass
class RandnParameterConfig(BaseParameterConfig):
    space: str = schema_utils.ProtectedString("randn")

    q: float = schema_utils.FloatRange(
        default=1,
        description=("Quantization factor. Output values will be rounded to the nearest multiple of `q` in range."),
    )


@DeveloperAPI
@register_parameter_type("randint")
@register_parameter_type("qrandint")
@register_parameter_type("lograndint")
@register_parameter_type("qlograndint")
@ludwig_dataclass
class RandintParameterConfig(BaseParameterConfig):
    space: str = schema_utils.ProtectedString("randint")

    lower: int = schema_utils.Integer(default=None, description="The minimum value the parameter can have.")

    upper: int = schema_utils.Integer(default=None, description="The maximum value the parameter can have.")

    q: int = schema_utils.Integer(
        default=1,
        description=("Quantization factor. Output values will be rounded to the nearest multiple of `q` in range."),
    )

    base: float = schema_utils.Float(default=None, allow_none=True, description="Logarithmic base.")
