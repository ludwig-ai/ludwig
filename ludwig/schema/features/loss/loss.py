from typing import List, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
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
from ludwig.schema.metadata import LOSS_METADATA


@DeveloperAPI
@dataclass(repr=False, order=True)
class BaseLossConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    type: str

    weight: float = 1.0


@DeveloperAPI
@dataclass(repr=False, order=True)
class MSELossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        MEAN_SQUARED_ERROR,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["MSELoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class MAELossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        MEAN_ABSOLUTE_ERROR,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["MAELoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class RMSELossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        ROOT_MEAN_SQUARED_ERROR,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["RMSELoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class RMSPELossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["RMSPELoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class BWCEWLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        BINARY_WEIGHTED_CROSS_ENTROPY,
        description="Type of loss.",
    )

    positive_class_weight: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Weight of the positive class.",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["positive_class_weight"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["confidence_penalty"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class SoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_weights"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["confidence_penalty"],
    )

    class_similarities: list = schema_utils.List(
        list,
        default=None,
        description="If not null this parameter is a c x c matrix in the form of a list of lists that contains the "
        "mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. ",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_similarities"],
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_similarities_temperature"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["weight"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class SequenceSoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SEQUENCE_SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["class_weights"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["confidence_penalty"],
    )

    class_similarities: list = schema_utils.List(
        list,
        default=None,
        description="If not null this parameter is a c x c matrix in the form of a list of lists that contains the "
        "mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. ",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["class_similarities"],
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description="",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["class_similarities_temperature"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["weight"],
    )

    unique: bool = schema_utils.Boolean(
        default=False,
        description="If true, the loss is only computed for unique elements in the sequence.",
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["unique"],
    )


@DeveloperAPI
@dataclass(repr=False, order=True)
class SigmoidCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SIGMOID_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
        parameter_metadata=LOSS_METADATA["SigmoidCrossEntropyLoss"]["class_weights"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["SigmoidCrossEntropyLoss"]["weight"],
    )
