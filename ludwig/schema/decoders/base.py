from abc import ABC

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ANOMALY, BINARY, CATEGORY, MODEL_ECD, MODEL_LLM, NUMBER, SET, TIMESERIES, VECTOR
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA


@DeveloperAPI
class BaseDecoderConfig(schema_utils.LudwigBaseConfig, ABC):
    """Base class for decoders."""

    type: str = schema_utils.StringOptions(
        ["regressor", "classifier", "projector", "generator", "tagger"],
        default=None,
        allow_none=True,
        description="The type of decoder to use.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["type"],
    )

    fc_layers: list[dict] = common_fields.FCLayersField()

    num_fc_layers: int = common_fields.NumFCLayersField(
        description="Number of fully-connected layers if `fc_layers` not specified."
    )

    fc_output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of fully connected stack.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_output_size"],
    )

    fc_use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector in the fc_stack.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_use_bias"],
    )

    fc_weights_initializer: str | dict = schema_utils.OneOfOptionsField(
        default="xavier_uniform",
        allow_none=True,
        description="The weights initializer to use for the layers in the fc_stack",
        field_options=[
            schema_utils.InitializerOptions(
                description="Preconfigured initializer to use for the layers in the fc_stack.",
                parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_weights_initializer"],
            ),
            schema_utils.Dict(
                description="Custom initializer to use for the layers in the fc_stack.",
                parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_weights_initializer"],
            ),
        ],
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_weights_initializer"],
    )

    fc_bias_initializer: str | dict = schema_utils.OneOfOptionsField(
        default="zeros",
        allow_none=True,
        description="The bias initializer to use for the layers in the fc_stack",
        field_options=[
            schema_utils.InitializerOptions(
                description="Preconfigured bias initializer to use for the layers in the fc_stack.",
                parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_bias_initializer"],
            ),
            schema_utils.Dict(
                description="Custom bias initializer to use for the layers in the fc_stack.",
                parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_bias_initializer"],
            ),
        ],
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_bias_initializer"],
    )

    fc_norm: str = common_fields.NormField()

    fc_norm_params: dict = common_fields.NormParamsField()

    fc_activation: str = schema_utils.ActivationOptions(default="relu")

    fc_dropout: float = common_fields.DropoutField()


@DeveloperAPI
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    @classmethod
    def module_name(cls):
        return "PassthroughDecoder"

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description="The passthrough decoder simply returns the raw numerical values coming from the combiner as "
        "outputs",
        parameter_metadata=DECODER_METADATA["PassthroughDecoder"]["type"],
    )

    input_size: int = schema_utils.PositiveInteger(
        default=1,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["PassthroughDecoder"]["input_size"],
    )


@DeveloperAPI
@register_decoder_config("regressor", [BINARY, NUMBER], model_types=[MODEL_ECD])
class RegressorConfig(BaseDecoderConfig):
    """RegressorConfig is a dataclass that configures the parameters used for a regressor decoder."""

    @classmethod
    def module_name(cls):
        return "Regressor"

    type: str = schema_utils.ProtectedString(
        "regressor",
        description=DECODER_METADATA["Regressor"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Regressor"]["input_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Regressor"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["bias_initializer"],
    )


@DeveloperAPI
@register_decoder_config("projector", [VECTOR, TIMESERIES], model_types=[MODEL_ECD])
class ProjectorConfig(BaseDecoderConfig):
    """ProjectorConfig is a dataclass that configures the parameters used for a projector decoder."""

    @classmethod
    def module_name(cls):
        return "Projector"

    type: str = schema_utils.ProtectedString(
        "projector",
        description=DECODER_METADATA["Projector"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Projector"]["input_size"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the output of the decoder.",
        parameter_metadata=DECODER_METADATA["Projector"]["output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Projector"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Projector"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Projector"]["bias_initializer"],
    )

    activation: str = schema_utils.ActivationOptions(
        default=None,
        description=" Indicates the activation function applied to the output.",
        parameter_metadata=DECODER_METADATA["Projector"]["activation"],
    )

    multiplier: float = schema_utils.FloatRange(
        default=1.0,
        min=0,
        min_inclusive=False,
        description=(
            "Multiplier to scale the activated outputs by. Useful when setting `activation` to something "
            "that outputs a value between [-1, 1] like tanh to re-scale values back to order of magnitude of "
            "the data you're trying to predict. A good rule of thumb in such cases is to pick a value like "
            "`x * (max - min)` where x is a scalar in the range [1, 2]. For example, if you're trying to predict "
            "something like temperature, it might make sense to pick a multiplier on the order of `100`."
        ),
    )

    clip: list[int] | tuple[int] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the output of the decoder to be within the given range.",
        parameter_metadata=DECODER_METADATA["Projector"]["clip"],
    )


@DeveloperAPI
@register_decoder_config("classifier", [CATEGORY, SET], model_types=[MODEL_ECD, MODEL_LLM])
class ClassifierConfig(BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "Classifier"

    type: str = schema_utils.ProtectedString(
        "classifier",
        description=DECODER_METADATA["Classifier"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Classifier"]["input_size"],
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of classes to predict.",
        parameter_metadata=DECODER_METADATA["Classifier"]["num_classes"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Classifier"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["bias_initializer"],
    )

    calibration: str | None = schema_utils.StringOptions(
        options=["temperature_scaling"],
        default=None,
        allow_none=True,
        description=(
            "Post-training calibration method to apply to the decoder logits. "
            "'temperature_scaling' learns a single scalar T that divides logits "
            "(calibrated_logits = logits / T) using NLL minimisation on the validation set. "
            "It never changes argmax predictions but improves probability reliability. "
            "See: Guo et al., 'On Calibration of Modern Neural Networks', ICML 2017. "
            "Set to null (default) to disable calibration."
        ),
    )

    mc_dropout_samples: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "Number of Monte Carlo forward passes to run at inference time with dropout enabled. "
            "When > 0, the decoder is run mc_dropout_samples times and the mean of the resulting "
            "probability distributions is used as the prediction; the variance across runs is reported "
            "as an 'uncertainty' tensor alongside 'predictions' and 'probabilities'. "
            "Setting this to 0 (default) disables MC Dropout. "
            "See: Gal & Ghahramani, 'Dropout as a Bayesian Approximation', ICML 2016."
        ),
    )


@DeveloperAPI
@register_decoder_config("mlp_classifier", [CATEGORY, BINARY], model_types=[MODEL_ECD])
class MLPClassifierConfig(BaseDecoderConfig):
    """Configuration for the MLPClassifier decoder.

    MLPClassifier stacks one or more fully-connected hidden layers (with configurable size,
    activation, and dropout) before the final linear projection to class logits. This is useful
    when the combiner output is too raw for a single-layer linear projection.

    When num_fc_layers=1 (the default), it applies a single hidden layer of size output_size
    before projecting to class logits. When num_fc_layers=0 the behaviour is equivalent to the
    standard Classifier. Increase num_fc_layers for more expressive capacity on harder
    classification problems.

    References:
        - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
          (for the calibration field).
        - Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing
          Model Uncertainty in Deep Learning", ICML 2016 (for mc_dropout_samples).
    """

    @classmethod
    def module_name(cls):
        return "MLPClassifier"

    type: str = schema_utils.ProtectedString(
        "mlp_classifier",
        description=(
            "Multi-layer perceptron classifier decoder. Stacks num_fc_layers fully-connected "
            "layers (each of size output_size) with activation and dropout, followed "
            "by a final linear projection to num_classes logits. "
            "Use this instead of the standard classifier when the combiner output benefits "
            "from additional non-linear transformation before the classification head."
        ),
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder. Set automatically from the combiner output size.",
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of classes to predict. Set automatically from the feature vocabulary size.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=1,
        description=(
            "Number of fully-connected hidden layers to stack before the classification head. "
            "When set to 1 (default) a single hidden layer of size output_size is applied. "
            "Set to 0 to make this decoder equivalent to the standard single-layer Classifier."
        ),
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of each hidden fully-connected layer. Only used when num_fc_layers > 0.",
    )

    activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="Activation function applied after each hidden fully-connected layer.",
    )

    dropout: float = common_fields.DropoutField()

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether each fully-connected layer (and the final projection) uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrices.",
        parameter_metadata=DECODER_METADATA["Classifier"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vectors.",
        parameter_metadata=DECODER_METADATA["Classifier"]["bias_initializer"],
    )

    calibration: str | None = schema_utils.StringOptions(
        options=["temperature_scaling"],
        default=None,
        allow_none=True,
        description=(
            "Post-training calibration method to apply to the decoder logits. "
            "'temperature_scaling' learns a single scalar T that divides logits "
            "(calibrated_logits = logits / T) using NLL minimisation on the validation set. "
            "It never changes argmax predictions but improves probability reliability. "
            "See: Guo et al., 'On Calibration of Modern Neural Networks', ICML 2017. "
            "Set to null (default) to disable calibration."
        ),
    )

    mc_dropout_samples: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "Number of Monte Carlo forward passes to run at inference time with dropout enabled. "
            "When > 0, the decoder is run mc_dropout_samples times and the mean of the resulting "
            "probability distributions is used as the prediction; the variance across runs is reported "
            "as an 'uncertainty' tensor alongside 'predictions' and 'probabilities'. "
            "Setting this to 0 (default) disables MC Dropout. "
            "See: Gal & Ghahramani, 'Dropout as a Bayesian Approximation', ICML 2016."
        ),
    )


@DeveloperAPI
@register_decoder_config("anomaly", [ANOMALY], model_types=[MODEL_ECD])
class AnomalyDecoderConfig(BaseDecoderConfig):
    """AnomalyDecoderConfig configures the anomaly decoder.

    The anomaly decoder computes ``||z - c||^2`` as the anomaly score, where ``z`` is
    the encoder/combiner output and ``c`` is the hypersphere center, which is initialized
    after the first epoch by computing the mean of all encoder outputs.

    This implements the geometric core of Deep SVDD (Ruff et al., ICML 2018).
    With Ludwig's ECD combiner you get free multimodal anomaly detection: any combination
    of tabular, image, text, or audio inputs is fused and mapped to the hypersphere.
    """

    @classmethod
    def module_name(cls):
        return "AnomalyDecoder"

    type: str = schema_utils.ProtectedString(
        "anomaly",
        description="Computes ||z - c||^2 as the anomaly score.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the encoder output. Set automatically from the FC stack output shape.",
    )
