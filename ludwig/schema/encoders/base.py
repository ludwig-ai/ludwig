from abc import ABC
from typing import Any, Dict, List, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_GBM, NUMBER, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.utils import register_encoder_config


@DeveloperAPI
@dataclass(repr=False, order=True)
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders."""

    type: str

    def get_fixed_preprocessing_params(self) -> Dict[str, Any]:
        """Returns a dict of fixed preprocessing parameters for the encoder if required."""
        return {}

    def is_pretrained(self) -> bool:
        return False


@DeveloperAPI
@register_encoder_config("passthrough", [NUMBER, VECTOR], model_types=[MODEL_ECD, MODEL_GBM])
@dataclass(order=True)
class PassthroughEncoderConfig(BaseEncoderConfig):
    """PassthroughEncoderConfig is a dataclass that configures the parameters used for a passthrough encoder."""

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description="Type of encoder.",
    )


@DeveloperAPI
@register_encoder_config("dense", [BINARY, NUMBER, VECTOR])
@dataclass(repr=False, order=True)
class DenseEncoderConfig(BaseEncoderConfig):
    """DenseEncoderConfig is a dataclass that configures the parameters used for a dense encoder."""

    type: str = schema_utils.ProtectedString(
        "dense",
        description="Type of encoder.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )

    activation: str = schema_utils.StringOptions(
        ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax"],
        default="relu",
        description="Activation function to apply to the output.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the dense encoder.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output of the feature.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    bias_initializer: Union[str, dict] = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
    )

    weights_initializer: Union[str, dict] = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        allow_none=True,
        default=None,
        description="Normalization to use in the dense layer.",
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters for normalization if norm is either batch or layer.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
    )

    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of fully connected layers to use in the encoder.",
    )
