from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import sequence_encoder_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    type: str = schema_utils.StringOptions(
        ["sequence"],
        default="sequence",
        allow_none=False,
        description="Type of combiner.",
    )

    main_sequence_feature: Optional[str] = schema_utils.String(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["main_sequence_feature"],
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["reduce_output"],
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(sequence_encoder_registry.keys()),
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["encoder"],
    )
