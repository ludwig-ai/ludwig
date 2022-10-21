from ludwig.schema import schema_utils as schema_utils


class BaseCombinerConfig(schema_utils.BaseMarshmallowConfig):
    """Base combiner config class."""

    type: str
