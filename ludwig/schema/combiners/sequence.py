from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD, SEQUENCE
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.sequence_concat import MAIN_SEQUENCE_FEATURE_DESCRIPTION
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


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
        allow_none=True,
        description=MAIN_SEQUENCE_FEATURE_DESCRIPTION,
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["main_sequence_feature"],
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=SEQUENCE,
        default="parallel_cnn",
        description="Encoder to apply to `main_sequence_feature`.",
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="Strategy to use to aggregate the embeddings of the items of the set.",
        parameter_metadata=COMBINER_METADATA["SequenceCombiner"]["reduce_output"],
    )
