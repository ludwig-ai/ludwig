from typing import ClassVar, Optional, List

from ludwig.schema import utils as schema_utils


class BaseInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base input feature config class."""

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


class BaseOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base output feature config class."""

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    dependencies: List[str] = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )


class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for input feature preprocessing. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding input feature class are copied over: check each class to check which attributes are different
    from the preprocessing of each feature.
    """

    feature_type: ClassVar[Optional[str]] = None
    "Class variable pointing to the corresponding preprocessor."
