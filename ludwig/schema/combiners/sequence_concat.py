from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass

MAIN_SEQUENCE_FEATURE_DESCRIPTION = """
Name of a sequence, text, or time series feature to concatenate the outputs
of the other features to. If no `main_sequence_feature` is specified, the combiner will look through all the features in
the order they are defined in the configuration and will look for a feature with a rank 3 tensor output (sequence, text
or time series). If it cannot find one it will raise an exception, otherwise the output of that feature will be used for
concatenating the other features along the sequence `s` dimension. If there are other input features with a rank 3
output tensor, the combiner will concatenate them alongside the `s` dimension. All sequence-like input features must
have identical `s` dimension, otherwise an error will be thrown.
"""


@DeveloperAPI
@register_combiner_config("sequence_concat")
@ludwig_dataclass
class SequenceConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence concat combiner."""

    @staticmethod
    def module_name():
        return "sequence_concat"

    type: str = schema_utils.ProtectedString(
        "sequence_concat",
        description=COMBINER_METADATA["sequence_concat"]["type"].long_description,
    )

    main_sequence_feature: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=MAIN_SEQUENCE_FEATURE_DESCRIPTION,
        parameter_metadata=COMBINER_METADATA["sequence_concat"]["main_sequence_feature"],
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="Strategy to use to aggregate the embeddings of the items of the set.",
        parameter_metadata=COMBINER_METADATA["sequence_concat"]["reduce_output"],
    )
