import logging
from typing import List

from marshmallow_dataclass import dataclass
from rich.console import Console

from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY, ParameterMetadata

logger = logging.getLogger(__name__)
_error_console = Console(stderr=True, style="bold red")
_info_console = Console(stderr=True, style="bold green")


@dataclass(repr=False)
class BaseFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    active: bool = True

    name: str = schema_utils.String(
        allow_none=True,
        description="Name of the feature.",
    )

    type: str = schema_utils.StringOptions(
        allow_none=True,
        options=[AUDIO, BAG, BINARY, CATEGORY, DATE, H3, IMAGE, NUMBER, SEQUENCE, SET, TEXT, TIMESERIES, VECTOR],
        description="Type of the feature.",
    )

    column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The column name of this feature. Defaults to name if not specified.",
    )

    proc_column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The name of the preprocessed column name of this feature. Internal only.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    def enable(self):
        """This function allows the user to specify which features from a dataset should be included during model
        training. This is the equivalent to toggling on a feature in the model creation UI.

        Returns:
            None
        """
        if self.active:
            _error_console.print("This feature is already enabled!")
        else:
            self.active = True
            _info_console.print(f"{self.name} feature enabled!\n")
            logger.info(self.__repr__())

    def disable(self):
        """This function allows the user to specify which features from a dataset should not be included during
        model training. This is the equivalent to toggling off a feature in the model creation UI.

        Returns:
            None
        """
        if not self.active:
            _error_console.print("This feature is already disabled!")
        else:
            self.active = False
            _info_console.print(f"{self.name} feature disabled!\n")
            logger.info(self.__repr__())


@dataclass(repr=False)
class BaseInputFeatureConfig(BaseFeatureConfig):
    """Base input feature config class."""

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@dataclass(repr=False)
class BaseOutputFeatureConfig(BaseFeatureConfig):
    """Base output feature config class."""

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    default_validation_metric: str = schema_utils.String(
        default=None,
        description="Internal only use parameter: default validation metric for output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: List[str] = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )
