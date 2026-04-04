from abc import ABC
from typing import TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_LLM, NUMBER, TEXT, TIMESERIES, VECTOR
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA

if TYPE_CHECKING:
    from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig


@DeveloperAPI
class BaseEncoderConfig(schema_utils.LudwigBaseConfig, ABC):
    """Base class for encoders."""

    type: str

    skip: bool = schema_utils.Boolean(
        False,
        "[internal] Whether to skip encoder and use input as output.",
        parameter_metadata=ENCODER_METADATA["BaseEncoder"]["skip"],
    )

    adapter: dict | None = schema_utils.Dict(
        default=None,
        allow_none=True,
        description=(
            "PEFT adapter configuration for parameter-efficient fine-tuning of pretrained encoders. "
            "Supports any adapter type registered in Ludwig (lora, vera, loha, etc.). "
            "Example: {type: lora, r: 8, alpha: 16, target_modules: [query, value]}. "
            "Only applicable to pretrained encoders (HuggingFace text encoders, TIMM image encoders)."
        ),
    )

    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "BasePreprocessingConfig"):
        pass

    def is_pretrained(self) -> bool:
        return False

    def can_cache_embeddings(self) -> bool:
        return False


@DeveloperAPI
@register_encoder_config("passthrough", [TEXT], model_types=[MODEL_LLM])
@register_encoder_config("passthrough", [BINARY, NUMBER, VECTOR], model_types=[MODEL_ECD])
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
@register_encoder_config("dense", [BINARY, NUMBER, VECTOR, TIMESERIES])
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

    bias_initializer: str | dict = common_fields.BiasInitializerField()

    weights_initializer: str | dict = common_fields.WeightsInitializerField()

    norm: str = common_fields.NormField()

    norm_params: dict = common_fields.NormParamsField()

    num_layers: int = common_fields.NumFCLayersField(default=1, non_zero=True)

    fc_layers: list[dict] = common_fields.FCLayersField()
