from ludwig.schema.model_utils import InputFeaturesContainer, OutputFeaturesContainer


class ModelMetadata:
    """
    Metadata class for internal only parameters used in the Ludwig Pipeline
    """

    def __init__(self, config_dict: dict):
        self.input_features: InputFeaturesContainer = InputFeaturesContainer()
        self.output_features: OutputFeaturesContainer = OutputFeaturesContainer()