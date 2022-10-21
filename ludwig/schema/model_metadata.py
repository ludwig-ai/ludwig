from ludwig.schema.model_utils import InputFeaturesContainer, OutputFeaturesContainer


class BaseInputFeatureInternalConfig(BaseFeatureConfig):
    """Base class for feature metadata."""

    column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The column name of this feature. Defaults to name if not specified.",
    )

    proc_column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The name of the preprocessed column name of this feature. Internal only.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )


class BaseOutputFeatureInternalConfig(BaseFeatureConfig):
    """Base class for feature metadata."""

    column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The column name of this feature. Defaults to name if not specified.",
    )

    proc_column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The name of the preprocessed column name of this feature. Internal only.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    default_validation_metric: str = schema_utils.String(
        default=None,
        description="Internal only use parameter: default validation metric for output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )


class ModelMetadata:
    """
    Metadata class for internal only parameters used in the Ludwig Pipeline
    """

    def __init__(self, config_dict: dict):
        self.input_features: InputFeaturesContainer = InputFeaturesContainer()
        self.output_features: OutputFeaturesContainer = OutputFeaturesContainer()