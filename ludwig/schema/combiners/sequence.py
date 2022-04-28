from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import sequence_encoder_registry


@dataclass
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    main_sequence_feature: Optional[str] = schema_utils.String(default=None, description="TODO: Document parameters.")

    reduce_output: Optional[str] = schema_utils.ReductionOptions(default=None, description="TODO: Document parameters.")

    encoder: Optional[str] = schema_utils.StringOptions(
        list(sequence_encoder_registry.keys()), default=None, description="TODO: Document parameters."
    )
