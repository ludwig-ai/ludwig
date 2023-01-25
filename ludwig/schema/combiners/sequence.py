from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SEQUENCE
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@ludwig_dataclass
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    @staticmethod
    def module_name():
        return "SequenceCombiner"

    type: str = schema_utils.ProtectedString(
        "sequence",
        description=COMBINER_METADATA["SequenceCombiner"]["type"].long_description,
    )

    main_sequence_feature: Optional[str] = schema_utils.String(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["main_sequence_feature"],
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SEQUENCE,
        default="parallel_cnn",
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["reduce_output"],
    )
