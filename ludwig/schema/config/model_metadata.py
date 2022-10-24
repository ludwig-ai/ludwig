from typing import List

from ludwig.constants import INPUT_FEATURES, NAME, OUTPUT_FEATURES, TYPE
from ludwig.schema.config.metadata_classes import InternalInputFeatureMetadata, InternalTrainerMetadata
from ludwig.schema.config.utils import (
    get_output_metadata_cls,
    InputFeaturesContainer,
    OutputFeaturesContainer,
    set_feature_column,
    set_proc_column,
)


class ModelMetadata:
    """Metadata class for internal only parameters used in the Ludwig Pipeline."""

    def __init__(self, config_dict: dict):
        self.input_features: InputFeaturesContainer = InputFeaturesContainer()
        self.output_features: OutputFeaturesContainer = OutputFeaturesContainer()
        self.trainer: InternalTrainerMetadata = InternalTrainerMetadata()

        set_feature_column(config_dict)
        set_proc_column(config_dict)
        self._initialize_input_features(config_dict[INPUT_FEATURES])
        self._initialize_output_features(config_dict[OUTPUT_FEATURES])

    @classmethod
    def from_dict(cls, dict_config):
        return cls(dict_config)

    def _initialize_input_features(self, feature_dicts: List[dict]) -> None:
        """This function initializes the input features on the ModelMetadata object that are specified in the user
        defined config dictionary.

        Args:
            feature_dicts: List of input feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelMetadata.
        """
        for feature_dict in feature_dicts:
            setattr(self.input_features, feature_dict[NAME], InternalInputFeatureMetadata())

    def _initialize_output_features(self, feature_dicts: List[dict]) -> None:
        """This function initializes the output features on the ModelMetadata object that are specified in the user
        defined config dictionary.

        Args:
            feature_dicts: List of output feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelMetadata.
        """
        for feature_dict in feature_dicts:
            # Retrieve input feature schema cls from registry to initialize feature
            feature_metadata_config = get_output_metadata_cls(feature_dict[TYPE])()

            # Assign feature on output features container
            setattr(self.output_features, feature_dict[NAME], feature_metadata_config)
