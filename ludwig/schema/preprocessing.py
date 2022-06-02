from typing import Optional, Union, List

from marshmallow_dataclass import dataclass
from marshmallow import Schema, fields, post_load

from ludwig.schema import utils as schema_utils


@dataclass
class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Base preprocessing config class."""

    force_split: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Force splitting of the input feature.",
    )

    split_probabilities: Optional[List[float]] = schema_utils.List(
        default=[0.7, 0.1, 0.2],
        allow_none=False,
        description="Probabilities for splitting the input data into train, validation, and test sets.",
    )

    stratify: Optional[bool] = schema_utils.StringOptions(
        ["TODO", "TODO"],
        default=None,
        description="Selection of categorical column to stratify the data on during data splitting.",
    )

    oversample_minority: Optional[float] = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="Random oversampling of the minority class during preprocessing.",
    )

    undersample_majority: Optional[float] = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="Random undersampling of the majority class during preprocessing.",
    )

    sample_ratio: Optional[float] = schema_utils.NonNegativeFloat(
        default=1.0,
        allow_none=False,
        description="Ratio of the data to sample during preprocessing.",
    )


@dataclass
class BinaryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryPreprocessingConfig is a dataclass that configures the parameters used for a binary input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_false", "fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_false",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary column",
    )

    fill_value: Union[int, float] = schema_utils.NumericOrStringOptionsField(
        ["yes", "YES", "Yes", "y", "Y", "true", "True", "TRUE", "t", "T", "1", "1.0", "no", "NO", "No", "n", "N",
         "false", "False", "FALSE", "f", "F", "0", "0.0"],
        allow_none=False,
        default=None,
        default_numeric=0,
        min=0,
        max=1,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    fallback_true_label: Optional[str] = schema_utils.NumericOrStringOptionsField(
        ["True", "False"],
        allow_none=True,
        default=None,
        default_numeric=1,
        default_option=None,
        min=0,
        max=1,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
                    "conventional boolean value"
    )
