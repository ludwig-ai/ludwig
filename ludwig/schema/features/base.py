from ludwig.schema import utils as schema_utils


class BaseInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base input feature config class."""

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


class BaseOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base output feature config class."""

    pass
