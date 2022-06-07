import json

from marshmallow_dataclass import dataclass
from marshmallow_jsonschema import JSONSchema as js

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig

# from typing import Optional


@dataclass
class TestConfig(BaseCombinerConfig):
    """Parameters for whatever."""

    size: int = schema_utils.TestOptions(
        default=32,
        description="`N_a` in the paper.",
        default_value_reasoning="some_reason",
        suggested_values=[1, 2],
        implications="this is important!",
        commonly_used=False,
    )


d = js().dump(TestConfig.Schema())
print(d)
print()
print(json.dumps(d))
