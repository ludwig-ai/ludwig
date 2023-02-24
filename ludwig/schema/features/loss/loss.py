from typing import Dict, List, Type, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    BINARY,
    BINARY_WEIGHTED_CROSS_ENTROPY,
    CATEGORY,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    NUMBER,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE,
    SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    SET,
    SIGMOID_CROSS_ENTROPY,
    SOFTMAX_CROSS_ENTROPY,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LOSS_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

ROBUST_LAMBDA_DESCRIPTION = (
    "Replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / c` where `c` is the number of "
    "classes. Useful in case of noisy labels."
)

CONFIDENCE_PENALTY_DESCRIPTION = (
    "Penalizes overconfident predictions (low entropy) by adding an additional term "
    "that penalizes too confident predictions by adding a `a * (max_entropy - entropy) / max_entropy` "
    "term to the loss, where a is the value of this parameter. Useful in case of noisy labels."
)

CLASS_WEIGHTS_DESCRIPTION = (
    "Weights to apply to each class in the loss. If not specified, all classes are weighted equally. "
    "The value can be a vector of weights, one for each class, that is multiplied to the "
    "loss of the datapoints that have that class as ground truth. It is an alternative to oversampling in "
    "case of unbalanced class distribution. The ordering of the vector follows the category to integer ID "
    "mapping in the JSON metadata file (the `<UNK>` class needs to be included too). Alternatively, the value "
    "can be a dictionary with class strings as keys and weights as values, like "
    "`{class_a: 0.5, class_b: 0.7, ...}`."
)

CLASS_SIMILARITIES_DESCRIPTION = (
    "If not `null` it is a `c x c` matrix in the form of a list of lists that contains the mutual similarity of "
    "classes. It is used if `class_similarities_temperature` is greater than 0. The ordering of the vector follows "
    "the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to be included too)."
)

CLASS_SIMILARITIES_TEMPERATURE_DESCRIPTION = (
    "The temperature parameter of the softmax that is performed on each row of `class_similarities`. The output of "
    "that softmax is used to determine the supervision vector to provide instead of the one hot vector that would be "
    "provided otherwise for each datapoint. The intuition behind it is that errors between similar classes are more "
    "tolerable than errors between really different classes."
)


@DeveloperAPI
@ludwig_dataclass
class BaseLossConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    type: str

    weight: float = 1.0

    @classmethod
    def name(cls) -> str:
        return "[undefined]"


_loss_registry = Registry[Type[BaseLossConfig]]()
_loss_feature_registry = Registry[Dict[str, Type[BaseLossConfig]]]()


@DeveloperAPI
def get_loss_schema_registry() -> Registry[Type[BaseLossConfig]]:
    return _loss_registry


@DeveloperAPI
def get_loss_cls(feature: str, name: str) -> Type[BaseLossConfig]:
    return _loss_feature_registry[feature][name]


@DeveloperAPI
def get_loss_classes(feature: str) -> Dict[str, Type[BaseLossConfig]]:
    return _loss_feature_registry[feature]


def register_loss(features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls: Type[BaseLossConfig]):
        _loss_registry[cls.type] = cls
        for feature in features:
            feature_registry = _loss_feature_registry.get(feature, {})
            feature_registry[cls.type] = cls
            _loss_feature_registry[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
@register_loss([NUMBER, TIMESERIES, VECTOR])
@ludwig_dataclass
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

    @classmethod
    def name(self) -> str:
        return "Mean Squared Error (MSE)"


@DeveloperAPI
@register_loss([NUMBER, TIMESERIES, VECTOR])
@ludwig_dataclass
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

    @classmethod
    def name(self) -> str:
        return "Mean Absolute Error (MAE)"


@DeveloperAPI
@register_loss([NUMBER])
@ludwig_dataclass
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

    @classmethod
    def name(self) -> str:
        return "Root Mean Squared Error (RMSE)"


@DeveloperAPI
@register_loss([NUMBER])
@ludwig_dataclass
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

    @classmethod
    def name(self) -> str:
        return "Root Mean Squared Percentage Error (RMSPE)"


@DeveloperAPI
@register_loss([BINARY])
@ludwig_dataclass
class BWCEWLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        BINARY_WEIGHTED_CROSS_ENTROPY,
        description="Type of loss.",
    )

    positive_class_weight: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="Weight of the positive class.",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["positive_class_weight"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=ROBUST_LAMBDA_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description=CONFIDENCE_PENALTY_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["confidence_penalty"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["BWCEWLoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Binary Weighted Cross Entropy (BWCE)"


@DeveloperAPI
@register_loss([CATEGORY, VECTOR])
@ludwig_dataclass
class SoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_weights"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=ROBUST_LAMBDA_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description=CONFIDENCE_PENALTY_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["confidence_penalty"],
    )

    class_similarities: list = schema_utils.List(
        list,
        default=None,
        description=CLASS_SIMILARITIES_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_similarities"],
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description=CLASS_SIMILARITIES_TEMPERATURE_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["class_similarities_temperature"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["SoftmaxCrossEntropyLoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Softmax Cross Entropy"


@DeveloperAPI
@register_loss([SEQUENCE, TEXT])
@ludwig_dataclass
class SequenceSoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SEQUENCE_SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["class_weights"],
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=ROBUST_LAMBDA_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["robust_lambda"],
    )

    confidence_penalty: float = schema_utils.NonNegativeFloat(
        default=0,
        description=CONFIDENCE_PENALTY_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["confidence_penalty"],
    )

    class_similarities: list = schema_utils.List(
        list,
        default=None,
        description=CLASS_SIMILARITIES_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SequenceSoftmaxCrossEntropyLoss"]["class_similarities"],
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description=CLASS_SIMILARITIES_TEMPERATURE_DESCRIPTION,
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

    @classmethod
    def name(self) -> str:
        return "Sequence Softmax Cross Entropy"


@DeveloperAPI
@register_loss([SET])
@ludwig_dataclass
class SigmoidCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SIGMOID_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.List(
        list_type=float,
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        parameter_metadata=LOSS_METADATA["SigmoidCrossEntropyLoss"]["class_weights"],
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["SigmoidCrossEntropyLoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Sigmoid Cross Entropy"
