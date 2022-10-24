from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass(repr=False)
class SequenceConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence concat combiner."""

    type: str = schema_utils.StringOptions(
        ["sequence_concat"],
        default="sequence_concat",
        allow_none=False,
        description="Type of combiner.",
    )

    main_sequence_feature: Optional[str] = schema_utils.String(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceConcatCombiner"]["main_sequence_feature"],
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceConcatCombiner"]["reduce_output"],
    )
