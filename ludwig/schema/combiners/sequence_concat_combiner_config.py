from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig


@dataclass
class SequenceConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence concat combiner."""

    main_sequence_feature: Optional[str] = utils.String(default=None, description="TODO: Document parameters.")

    reduce_output: Optional[str] = utils.ReductionOptions(default=None, description="TODO: Document parameters.")
