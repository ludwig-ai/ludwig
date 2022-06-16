from ludwig.schema import utils as schema_utils


class BaseFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base input feature config class."""

    name: str = schema_utils.String(
        default=None,
        description="Name of the feature.",
    )

    type: str = schema_utils.String(
        default=None,
        description="Type of the feature.",
    )
