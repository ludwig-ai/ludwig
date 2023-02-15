from abc import ABC
from typing import Any, Dict, List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_GBM, NUMBER, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.initializers import (
    BiasInitializerDataclassField,
    InitializerConfig,
    WeightsInitializerDataclassField,
)
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders."""

    type: str

    skip: bool = schema_utils.Boolean(
        False,
        "[internal] Whether to skip encoder and use input as output.",
        parameter_metadata=ENCODER_METADATA["BaseEncoder"]["skip"],
    )

    def get_fixed_preprocessing_params(self) -> Dict[str, Any]:
        return {}

    def is_pretrained(self) -> bool:
        return False

    def can_cache_embeddings(self) -> bool:
        return False


@DeveloperAPI
@register_encoder_config("passthrough", [BINARY, NUMBER, VECTOR], model_types=[MODEL_ECD, MODEL_GBM])
@ludwig_dataclass
class PassthroughEncoderConfig(BaseEncoderConfig):
    """PassthroughEncoderConfig is a dataclass that configures the parameters used for a passthrough encoder."""

    @staticmethod
    def module_name():
        return "PassthroughEncoder"

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description=ENCODER_METADATA["PassthroughEncoder"]["type"].long_description,
    )


@DeveloperAPI
@register_encoder_config("dense", [BINARY, NUMBER, VECTOR])
@ludwig_dataclass
class DenseEncoderConfig(BaseEncoderConfig):
    """DenseEncoderConfig is a dataclass that configures the parameters used for a dense encoder."""

    @staticmethod
    def module_name():
        return "DenseEncoder"

    type: str = schema_utils.ProtectedString(
        "dense",
        description=ENCODER_METADATA["DenseEncoder"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["dropout"],
    )

    activation: str = schema_utils.StringOptions(
        ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax"],
        default="relu",
        description="Activation function to apply to the output.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["activation"],
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the dense encoder.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["input_size"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output of the feature.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["use_bias"],
    )

    bias_initializer: InitializerConfig = BiasInitializerDataclassField(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["bias_initializer"],
    )

    weights_initializer: InitializerConfig = WeightsInitializerDataclassField(
        description="Initializer for the weight matrix.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["weights_initializer"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        allow_none=True,
        default=None,
        description="Normalization to use in the dense layer.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters for normalization if norm is either batch or layer.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["norm_params"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["num_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of fully connected layers to use in the encoder.",
        parameter_metadata=ENCODER_METADATA["DenseEncoder"]["fc_layers"],
    )
