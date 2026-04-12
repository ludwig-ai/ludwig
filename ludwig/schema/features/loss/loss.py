from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    ANOMALY,
    BINARY,
    BINARY_WEIGHTED_CROSS_ENTROPY,
    CATEGORY,
    CORN,
    DEEP_SAD,
    DEEP_SVDD,
    DICE_LOSS,
    DROCC,
    ENTMAX15_LOSS,
    ENTROPIC_OPEN_SET,
    FOCAL_LOSS,
    HUBER,
    IMAGE,
    LOVASZ_SOFTMAX_LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    MEAN_SQUARED_ERROR,
    NEXT_TOKEN_SOFTMAX_CROSS_ENTROPY,
    NT_XENT_LOSS,
    NUMBER,
    OBJECTOSPHERE,
    POLY_LOSS,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE,
    SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    SET,
    SIGMOID_CROSS_ENTROPY,
    SOFTMAX_CROSS_ENTROPY,
    SPARSEMAX_LOSS,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LOSS_METADATA
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
class BaseLossConfig(schema_utils.LudwigBaseConfig):
    """Base class for feature configs."""

    type: str

    weight: float = 1.0

    @classmethod
    def name(cls) -> str:
        return "[undefined]"


_loss_registry = Registry[type[BaseLossConfig]]()
_loss_feature_registry = Registry[dict[str, type[BaseLossConfig]]]()


@DeveloperAPI
def get_loss_schema_registry() -> Registry[type[BaseLossConfig]]:
    return _loss_registry


@DeveloperAPI
def get_loss_cls(feature: str, name: str) -> type[BaseLossConfig]:
    return _loss_feature_registry[feature][name]


@DeveloperAPI
def get_loss_classes(feature: str) -> dict[str, type[BaseLossConfig]]:
    return _loss_feature_registry[feature]


def register_loss(features: str | list[str]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls: type[BaseLossConfig]):
        _loss_registry[cls.type] = cls
        for feature in features:
            feature_registry = _loss_feature_registry.get(feature, {})
            feature_registry[cls.type] = cls
            _loss_feature_registry[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
@register_loss([NUMBER, TIMESERIES, VECTOR])
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
@register_loss([NUMBER, TIMESERIES, VECTOR])
class MAPELossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        MEAN_ABSOLUTE_PERCENTAGE_ERROR,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["MAELoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Mean Absolute Percentage Error (MAPE)"


@DeveloperAPI
@register_loss([NUMBER])
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
@register_loss([CATEGORY, VECTOR, IMAGE])
class SoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: list[float] | dict | None = schema_utils.OneOfOptionsField(
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        field_options=[
            schema_utils.Dict(default=None, allow_none=True),
            schema_utils.List(list_type=float, allow_none=False),
        ],
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
class SequenceSoftmaxCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SEQUENCE_SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: list[float] | dict | None = schema_utils.OneOfOptionsField(
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        field_options=[
            schema_utils.Dict(default=None, allow_none=True),
            schema_utils.List(list_type=float, allow_none=False),
        ],
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
@register_loss([SEQUENCE, TEXT])
class NextTokenSoftmaxCrossEntropyLossConfig(SequenceSoftmaxCrossEntropyLossConfig):
    type: str = schema_utils.ProtectedString(
        NEXT_TOKEN_SOFTMAX_CROSS_ENTROPY,
        description="Type of loss.",
    )

    @classmethod
    def name(self) -> str:
        return "Next Token Softmax Cross Entropy"


@DeveloperAPI
@register_loss([SET])
class SigmoidCrossEntropyLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        SIGMOID_CROSS_ENTROPY,
        description="Type of loss.",
    )

    class_weights: list[float] | dict | None = schema_utils.OneOfOptionsField(
        default=None,
        description=CLASS_WEIGHTS_DESCRIPTION,
        field_options=[
            schema_utils.Dict(default=None, allow_none=True),
            schema_utils.List(list_type=float, allow_none=False),
        ],
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


@DeveloperAPI
@register_loss([NUMBER, TIMESERIES, VECTOR])
class HuberLossConfig(BaseLossConfig):
    type: str = schema_utils.ProtectedString(
        HUBER,
        description=(
            "Loss that combines advantages of both `mean_absolute_error` (MAE) and `mean_squared_error` (MSE). The "
            "delta-scaled L1 region makes the loss less sensitive to outliers than MSE, while the L2 region provides "
            "smoothness over MAE near 0. See [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) for more details."
        ),
    )

    delta: float = schema_utils.FloatRange(
        default=1.0,
        min=0,
        min_inclusive=False,
        description="Threshold at which to change between delta-scaled L1 and L2 loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["MSELoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Huber Loss"


@DeveloperAPI
@register_loss([CATEGORY])
class CORNLossConfig(BaseLossConfig):
    """Conditional Ordinal Regression for Neural networks, used for ordered cateogry values.

    Source:
    Xintong Shi, Wenzhi Cao, and Sebastian Raschka (2021).
    Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities.
    Arxiv preprint; https://arxiv.org/abs/2111.08851
    """

    type: str = schema_utils.ProtectedString(
        CORN,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
        parameter_metadata=LOSS_METADATA["MSELoss"]["weight"],
    )

    @classmethod
    def name(self) -> str:
        return "Conditional Ordinal Regression (CORN)"

    @property
    def class_weights(self) -> int:
        return 1.0

    @property
    def class_similarities_temperature(self) -> int:
        return 0


@DeveloperAPI
@register_loss([ANOMALY])
class DeepSVDDLossConfig(BaseLossConfig):
    """Deep Support Vector Data Description (Deep SVDD) loss for anomaly detection.

    Trains the encoder to map normal data into a compact hypersphere centred at c.
    Hard-boundary objective: L = mean(||z - c||^2) for all training points.
    Soft-boundary (nu > 0): L = R + (1/nu) * mean(max(0, ||z - c||^2 - R)) where
    R is the nu-th quantile of distances (no gradient through R).

    Reference: Ruff et al., "Deep One-Class Classification", ICML 2018.
    """

    type: str = schema_utils.ProtectedString(
        DEEP_SVDD,
        description="Deep SVDD loss — pulls encoder outputs toward hypersphere center c.",
    )

    nu: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        min_inclusive=False,
        description=(
            "Fraction of training examples allowed outside the hypersphere (soft-boundary mode). "
            "Set nu=0 for hard-boundary SVDD where all points are pulled toward c."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Deep SVDD"


@DeveloperAPI
@register_loss([ANOMALY])
class DeepSADLossConfig(BaseLossConfig):
    """Deep Semi-supervised Anomaly Detection (Deep SAD) loss.

    Extends Deep SVDD with labeled anomaly examples. Normal/unlabeled samples
    (target != 1) are pulled toward center c; labeled anomalies (target == 1)
    are pushed away via an inverted distance term weighted by eta.

    Reference: Ruff et al., "Deep Semi-Supervised Anomaly Detection", ICLR 2020.
    """

    type: str = schema_utils.ProtectedString(
        DEEP_SAD,
        description="Deep SAD loss — semi-supervised, labeled anomalies pushed away from center c.",
    )

    eta: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight for the labeled anomaly repulsion term.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Deep SAD"


@DeveloperAPI
@register_loss([ANOMALY])
class DROCCLossConfig(BaseLossConfig):
    """Deeply Robust One-Class Classification (DROCC) loss.

    Prevents hypersphere collapse (all representations converge to c) via
    an adversarial perturbation regulariser. Recommended for expressive encoders
    (e.g. transformers) that are prone to degenerate solutions.

    Reference: Goyal et al., "DROCC: Deep Robust One-Class Classification", ICML 2020.
    """

    type: str = schema_utils.ProtectedString(
        DROCC,
        description="DROCC loss — prevents collapse via adversarial score perturbations.",
    )

    perturbation_strength: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="Magnitude of adversarial perturbations. Typical range: 0.01–0.5.",
    )

    num_perturbation_steps: int = schema_utils.PositiveInteger(
        default=5,
        description="Gradient ascent steps for adversarial perturbation generation.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "DROCC"


@DeveloperAPI
@register_loss([BINARY, CATEGORY])
class EntropicOpenSetLossConfig(BaseLossConfig):
    """Entropic Open-Set Loss for open-set recognition.

    Combines standard cross-entropy for known-class samples with an entropy
    maximisation term for background/unknown samples. This discourages the
    network from making confident predictions on out-of-distribution inputs,
    "curing" network agnostophobia.

    The background class (identified by ``background_class``) is treated as the
    catch-all unknown category. Samples with that label contribute only the
    entropic term; all other samples contribute cross-entropy as normal.
    If ``background_class`` is None the loss reduces to standard cross-entropy.

    Reference:
        Dhamija et al., "Reducing Network Agnostophobia", NeurIPS 2018.
        https://arxiv.org/abs/1811.04110
    """

    type: str = schema_utils.ProtectedString(
        ENTROPIC_OPEN_SET,
        description=(
            "Entropic open-set loss — cross-entropy for known classes + entropy "
            "maximisation for the background/unknown class."
        ),
    )

    background_class: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Class index that represents 'unknown' or background samples. "
            "Samples with this label receive the entropic penalty instead of "
            "cross-entropy. Set to None to disable open-set behaviour and fall "
            "back to standard cross-entropy."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Entropic Open-Set"


@DeveloperAPI
@register_loss([BINARY, CATEGORY])
class ObjectosphereLossConfig(BaseLossConfig):
    """Objectosphere Loss for open-set recognition.

    Extends the Entropic Open-Set Loss with a feature-magnitude constraint:

    * **Known samples**: standard cross-entropy + hinge term that pushes the
      logit L2 norm above ``xi`` (large magnitude → confident, well-separated
      representations).
    * **Unknown/background samples**: entropy maximisation + magnitude
      minimisation weighted by ``zeta`` (small magnitude → low-confidence,
      "don't know" responses).

    The combined objective makes it easy to threshold on logit magnitude at
    inference time: known-class inputs will have large norms; truly unknown
    inputs will have small norms regardless of the argmax prediction.

    Reference:
        Dhamija et al., "Reducing Network Agnostophobia", NeurIPS 2018.
        https://arxiv.org/abs/1811.04110
    """

    type: str = schema_utils.ProtectedString(
        OBJECTOSPHERE,
        description=(
            "Objectosphere loss — cross-entropy + magnitude push for known classes, "
            "entropy maximisation + magnitude suppression for unknowns."
        ),
    )

    background_class: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Class index that represents 'unknown' or background samples. "
            "Samples with this label receive the entropic + magnitude-suppression "
            "penalty. Set to None to fall back to standard cross-entropy."
        ),
    )

    xi: float = schema_utils.NonNegativeFloat(
        default=10.0,
        description=(
            "Minimum desired logit L2 norm for known-class samples. "
            "A hinge term max(0, xi - ||z||)² is added to push representations "
            "of known inputs above this threshold. Typical values: 1–50."
        ),
    )

    zeta: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description=(
            "Weight applied to the magnitude-suppression term for unknown samples. "
            "Higher values more aggressively shrink logit norms for background inputs."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Objectosphere"


@DeveloperAPI
@register_loss([BINARY, CATEGORY, IMAGE])
class FocalLossConfig(BaseLossConfig):
    """Focal Loss for classification with class imbalance.

    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy loss
    so that easy examples contribute less to the gradient.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    type: str = schema_utils.ProtectedString(
        FOCAL_LOSS,
        description="Type of loss.",
    )

    alpha: float = schema_utils.FloatRange(
        default=0.25,
        min=0.0,
        max=1.0,
        description=(
            "Weighting factor for the positive class in binary classification. "
            "Balances the importance of positive/negative examples."
        ),
    )

    gamma: float = schema_utils.NonNegativeFloat(
        default=2.0,
        description=(
            "Focusing parameter that reduces the loss contribution from easy examples "
            "and extends the range in which an example receives low loss. "
            "gamma=0 reduces focal loss to standard cross-entropy."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Focal Loss"


@DeveloperAPI
@register_loss([IMAGE])
class DiceLossConfig(BaseLossConfig):
    """Dice Loss for image segmentation.

    Computes 1 minus the Dice coefficient between predicted soft masks and
    one-hot ground-truth masks.

    Reference: Milletari et al., "V-Net", 3DV 2016.
    """

    type: str = schema_utils.ProtectedString(
        DICE_LOSS,
        description="Type of loss.",
    )

    smooth: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description=(
            "Laplace smoothing term added to numerator and denominator to prevent "
            "division by zero when both prediction and target are empty."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Dice Loss"


@DeveloperAPI
@register_loss([IMAGE])
class LovaszSoftmaxLossConfig(BaseLossConfig):
    """Lovasz-Softmax Loss for multi-class semantic segmentation.

    Uses the Lovasz extension of submodular functions to construct a convex
    surrogate for the per-class IoU loss.

    Reference: Berman et al., "The Lovasz-Softmax Loss", CVPR 2018.
    """

    type: str = schema_utils.ProtectedString(
        LOVASZ_SOFTMAX_LOSS,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Lovász-Softmax Loss"


@DeveloperAPI
@register_loss([VECTOR])
class NTXentLossConfig(BaseLossConfig):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss (SimCLR).

    Given a batch of N vector representations, computes contrastive loss
    assuming consecutive pairs (2i, 2i+1) are positive pairs.

    Reference: Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020.
    """

    type: str = schema_utils.ProtectedString(
        NT_XENT_LOSS,
        description="Type of loss.",
    )

    temperature: float = schema_utils.FloatRange(
        default=0.07,
        min=0.0,
        min_inclusive=False,
        description=(
            "Temperature parameter for scaling the cosine similarity scores. "
            "Lower values make the distribution sharper, higher values softer."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "NT-Xent Loss"


@DeveloperAPI
@register_loss([CATEGORY])
class PolyLossConfig(BaseLossConfig):
    """PolyLoss for multi-class classification.

    Extends cross-entropy with a first-order polynomial correction term
    epsilon * (1 - p_t) that upweights hard examples.

    Reference: Leng et al., "PolyLoss", ICLR 2022.
    """

    type: str = schema_utils.ProtectedString(
        POLY_LOSS,
        description="Type of loss.",
    )

    epsilon: float = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        description=(
            "Coefficient for the polynomial correction term. " "epsilon=0 reduces PolyLoss to standard cross-entropy."
        ),
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Poly Loss"


@DeveloperAPI
@register_loss([CATEGORY, TEXT, SEQUENCE])
class SparsemaxLossConfig(BaseLossConfig):
    """Sparsemax Loss: a sparse alternative to softmax cross-entropy.

    The natural loss companion to the sparsemax activation. Assigns zero
    gradient to classes outside the sparsemax support.

    Reference: Martins & Astudillo, "From Softmax to Sparsemax", ICML 2016.
    """

    type: str = schema_utils.ProtectedString(
        SPARSEMAX_LOSS,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Sparsemax Loss"


@DeveloperAPI
@register_loss([CATEGORY, TEXT, SEQUENCE])
class Entmax15LossConfig(BaseLossConfig):
    """Entmax-1.5 Loss: a semi-sparse alternative to softmax cross-entropy.

    The Fenchel-conjugate loss of the alpha=1.5 entmax activation. Produces
    moderately sparse probability distributions between softmax and sparsemax.

    Reference: Peters et al., "Sparse Sequence-to-Sequence Models", ACL 2019.
    """

    type: str = schema_utils.ProtectedString(
        ENTMAX15_LOSS,
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )

    @classmethod
    def name(cls) -> str:
        return "Entmax-1.5 Loss"
