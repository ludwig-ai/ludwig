from marshmallow_dataclass import dataclass

from ludwig.combiners.combiners import BaseCombinerConfig
from ludwig.utils.marshmallow_schema_utils import BaseMarshmallowConfig


@dataclass
class CustomTestCombinerConfig(BaseCombinerConfig):
    foo: bool = False


@dataclass
class CustomTestSchema(BaseMarshmallowConfig):
    """sample docstring."""

    foo: int = 5
    "foo (default: 5)"
