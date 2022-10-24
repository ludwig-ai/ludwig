import yaml

from ludwig.constants import ACTIVE, COLUMN, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PROC_COLUMN, TYPE
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.schema.utils import convert_submodules
from ludwig.utils.registry import Registry

internal_output_config_registry = Registry()


def get_output_metadata_cls(name: str):
    return internal_output_config_registry[name]


class BaseFeatureContainer:
    """Base Feature container for input and output features."""

    def to_dict(self):
        """Method for getting a dictionary representation of the input features.

        Returns:
            Dictionary of input features specified.
        """
        return convert_submodules(self.__dict__)

    def to_list(self):
        """Method for getting a list representation of the input features.

        Returns:
            List of input features specified.
        """
        return list(convert_submodules(self.__dict__).values())

    def filter_features(self):
        """This function is intended to filter out the parameters on input/output features that we want to show in
        the config object repr."""
        return {
            key: {k: v for k, v in value.items() if k in {NAME, TYPE, ACTIVE}} for key, value in self.to_dict().items()
        }

    def get(self, feature_name):
        """Gets a feature by name.

        raises AttributeError if no feature with the specified name is present.
        """
        return getattr(self, feature_name)

    def __repr__(self):
        filtered_repr = self.filter_features()
        return yaml.dump(filtered_repr, sort_keys=True)


class InputFeaturesContainer(BaseFeatureContainer):
    """InputFeatures is a container for all input features."""

    pass


class OutputFeaturesContainer(BaseFeatureContainer):
    """OutputFeatures is a container for all output features."""

    pass


def set_feature_column(config: dict) -> None:
    for feature in config[INPUT_FEATURES] + config[OUTPUT_FEATURES]:
        if COLUMN not in feature:
            feature[COLUMN] = feature[NAME]


def set_proc_column(config: dict) -> None:
    for feature in config[INPUT_FEATURES] + config[OUTPUT_FEATURES]:
        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)