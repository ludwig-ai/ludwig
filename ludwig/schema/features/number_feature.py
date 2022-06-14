from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import NumberPreprocessingConfig
from ludwig.schema.features.utils import register_input_feature, register_output_feature


@dataclass
@register_input_feature("number")
class NumberInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """NumberInputFeature is a dataclass that configures the parameters used for a number input feature."""

    name: str = schema_utils.String(
        default=None,
        allow_none=False,
        description="Name of the feature. Must be unique within the model.",
    )

    type: str = "number"

    preprocessing: Optional[str] = NumberPreprocessingConfig(
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense"],
        default="passthrough",
        description="Encoder to use for this number feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
@register_output_feature("number")
class NumberOutputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):

    name: str = schema_utils.String(
        default=None,
        allow_none=False,
        description="Name of the feature. Must be unique within the model.",
    )

    type: str = "number"
