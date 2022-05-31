from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import COMBINED, LOSS, TRAINING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base


@dataclass
class BinaryFeataure(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """BinaryFeataure is a dataclass that configures the hyperparameters used for a binary input feature."""

    preprocessing: Optional[str] = schema_utils.StringOptions(
        list(base.preprocessing_registry.keys()),
        default=None,
        description="",
    )

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
        min=0,
        max=1,
    )


