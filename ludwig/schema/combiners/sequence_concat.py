from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@dataclass(repr=False)
class SequenceConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence concat combiner."""

    type: str = schema_utils.ProtectedString(
        "sequence_concat",
        description="The sequence_concat combiner assumes at least one output from the encoders is a tensor of size "
        "`b x s x h` where `b` is the batch size, `s` is the length of the sequence and `h` is the hidden "
        "dimension. A sequence-like (sequence, text or time series) input feature can be specified with "
        "the `main_sequence_feature` parameter which takes the name of sequence-like input feature as its "
        "value. If no `main_sequence_feature` is specified, the combiner will look through all the "
        "features in the order they are defined in the configuration and will look for a feature with a "
        "rank 3 tensor output (sequence, text or time series). If it cannot find one it will raise an "
        "exception, otherwise the output of that feature will be used for concatenating the other features "
        "along the sequence `s` dimension."
        ""
        "If there are other input features with a rank 3 output tensor, the combiner will concatenate "
        "them alongside the s dimension, which means that all of them must have identical s dimension, "
        "otherwise a dimension mismatch error will be returned thrown during training when a datapoint "
        "with two sequential features of different lengths are provided. "
        ""
        "Other features that have a b x h rank 2 tensor output will be replicated s times and "
        "concatenated to the s dimension. The final output is a b x s x h' tensor where h' is the size of "
        "the concatenation of the h dimensions of all input features. ",
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
