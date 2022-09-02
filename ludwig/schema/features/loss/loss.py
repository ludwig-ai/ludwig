from typing import List, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import (
    BINARY_WEIGHTED_CROSS_ENTROPY,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    SIGMOID_CROSS_ENTROPY,
    SOFTMAX_CROSS_ENTROPY,
)
from ludwig.schema import utils as schema_utils


@dataclass
class BaseLossConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    type: str

    weight: float = 1.0


@dataclass
class MSELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[MEAN_SQUARED_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class MAELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[MEAN_ABSOLUTE_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class RMSELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[ROOT_MEAN_SQUARED_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class RMSPELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[ROOT_MEAN_SQUARED_PERCENTAGE_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class BWCEWLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[BINARY_WEIGHTED_CROSS_ENTROPY],
        description="Type of loss.",
    )

    positive_class_weight: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Weight of the positive class.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(default=0, description="")

    confidence_penalty: float = schema_utils.NonNegativeFloat(default=0, description="")

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class SoftmaxCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SOFTMAX_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(default=0, description="")

    confidence_penalty: float = schema_utils.NonNegativeFloat(default=0, description="")

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(default=0, description="")

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class SequenceSoftmaxCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SEQUENCE_SOFTMAX_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(default=0, description="")

    confidence_penalty: float = schema_utils.NonNegativeFloat(default=0, description="")

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(default=0, description="")

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    unique: bool = schema_utils.Boolean(
        default=False,
        description="If true, the loss is only computed for unique elements in the sequence.",
    )


@dataclass
class SigmoidCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SIGMOID_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )
