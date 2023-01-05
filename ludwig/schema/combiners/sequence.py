from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SEQUENCE
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@dataclass(repr=False, order=True)
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    type: str = schema_utils.ProtectedString(
        "sequence",
        description="The sequence combiner stacks a sequence concat combiner with a sequence encoder. All the "
        "considerations about input tensor ranks described for the sequence concat combiner apply also in "
        "this case, but the main difference is that this combiner uses the `b x s x h` output of the "
        "sequence concat combiner, where `b` is the batch size, `s` is the sequence length and `h` is the "
        "sum of the hidden dimensions of all input features, as input for any of the sequence encoders "
        "described in the sequence features encoders section. All considerations on the shape of "
        "the outputs for the sequence encoders also apply to the sequence combiner.",
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
