from abc import ABC
from typing import Dict, List, Tuple, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, MODEL_ECD, MODEL_GBM, MODEL_LLM, NUMBER, SET, TIMESERIES, VECTOR
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders."""

    type: str = schema_utils.StringOptions(
        ["regressor", "classifier", "projector", "generator", "tagger"],
        default=None,
        allow_none=True,
        description="The type of decoder to use.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["type"],
    )

    fc_layers: List[dict] = common_fields.FCLayersField()

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

    fc_weights_initializer: Union[str, Dict] = schema_utils.OneOfOptionsField(
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

    fc_bias_initializer: Union[str, Dict] = schema_utils.OneOfOptionsField(
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
@ludwig_dataclass
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
@register_decoder_config("regressor", [BINARY, NUMBER], model_types=[MODEL_ECD, MODEL_GBM])
@ludwig_dataclass
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
@ludwig_dataclass
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

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the output of the decoder to be within the given range.",
        parameter_metadata=DECODER_METADATA["Projector"]["clip"],
    )


@DeveloperAPI
@register_decoder_config("classifier", [CATEGORY, SET], model_types=[MODEL_ECD, MODEL_GBM, MODEL_LLM])
@ludwig_dataclass
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


@DeveloperAPI
@register_decoder_config("tart", [BINARY, CATEGORY], model_types=[MODEL_ECD, MODEL_GBM, MODEL_LLM])
@ludwig_dataclass
class TARTDecoderConfig(BaseDecoderConfig):
    reasoning_head: str = schema_utils.StringOptions(
        options=[
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-125m",
            "EleutherAI/pythia-1.4b-deduped",
            "EleutherAI/pythia-2.8b-deduped",
            "facebook/opt-iml-max-1.3b",
        ],
        default=None,
        allow_none=True,
        description="The model architecture to use in the reasoning head. Defaults to the GPT-2 backbone.",
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField(default=256)

    num_pca_components: int = schema_utils.PositiveInteger(
        default=16, description="Target size of the PCA projections."
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256, description="Size of the hidden state of the reasoning head."
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=12, description="Number of layers in the reasoning head model. Only used with the GPT-2 backbone."
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="The number of attention heads in the reasoning module. Only used with the GPT-2 backbone.",
    )

    embedding_protocol: str = schema_utils.StringOptions(
        options=["vanilla", "loo"],
        default="loo",
        description=(
            'The embedding protocol used to create TART inputs. For "vanilla", the test example is appended to the '
            'training set and the sequence is passed to the base model. For "loo" embeddings are generated for each '
            "train example separately by placing all other train examples before it in the prompt and averaging the "
            "embeddings over the final example's tokens."
        ),
    )
