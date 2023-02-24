from abc import ABC
from typing import Any, Dict, List, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_GBM, NUMBER, VECTOR
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.utils import register_encoder_config
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

    dropout: float = common_fields.DropoutField()

    activation: str = schema_utils.ActivationOptions()

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

    bias_initializer: Union[str, dict] = common_fields.BiasInitializerField()

    weights_initializer: Union[str, dict] = common_fields.WeightsInitializerField()

    norm: str = common_fields.NormField()

    norm_params: dict = common_fields.NormParamsField()

    num_layers: int = common_fields.NumFCLayersField(default=1)

    fc_layers: List[dict] = common_fields.FCLayersField()
