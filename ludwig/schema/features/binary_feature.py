from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BinaryPreprocessingConfig

from marshmallow import Schema, fields, post_load, ValidationError


@dataclass
class BinaryInputFeature(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """BinaryInputFeature is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: Optional[str] = BinaryPreprocessingConfig(
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense"],
        default="passthrough",
        description="Encoder to use for this binary feature.",
    )

    tied: Optional[str] = schema_utils.StringOptions(  # TODO: Get input features used
        ["TODO", "TODO"],
        default=None,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )

    @post_load(pass_original=True)
    def add_baz_to_bar(self, data, original_data, **kwargs):
        baz = original_data.get("baz")
        if baz:
            data["bar"] = data["bar"] + baz
        return data


    num_layers: Optional[int] = schema_utils.Posti(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
    )

    output_size: Optional[int] = schema_utils.Integer(
        default=256,
        description="Size of the output of the feature.",
    )

@dataclass
class BinaryOutputFeature(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """BinaryOutputFeature is a dataclass that configures the parameters used for a binary output feature."""
