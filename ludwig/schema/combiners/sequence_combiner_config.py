from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.combiners.utils import sequence_encoder_registry
from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig


@dataclass
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    main_sequence_feature: Optional[str] = utils.String(default=None, description="TODO: Document parameters.")

    reduce_output: Optional[str] = utils.ReductionOptions(default=None, description="TODO: Document parameters.")

    encoder: Optional[str] = utils.StringOptions(
        list(sequence_encoder_registry.keys()), default=None, description="TODO: Document parameters."
    )
