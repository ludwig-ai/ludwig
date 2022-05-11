from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig


@dataclass
class SequenceConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence concat combiner."""

    main_sequence_feature: Optional[str] = schema_utils.String(default=None, description="")

    reduce_output: Optional[str] = schema_utils.ReductionOptions(default=None, description="")
