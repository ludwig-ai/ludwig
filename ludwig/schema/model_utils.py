import yaml

from ludwig.constants import (
    ACTIVE,
    BINARY,
    CATEGORY,
    COLUMN,
    COMBINER,
    DECODER,
    DEFAULT_VALIDATION_METRIC,
    DEFAULTS,
    ENCODER,
    EXECUTOR,
    HYPEROPT,
    INPUT_FEATURES,
    LOSS,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NAME,
    NUMBER,
    OPTIMIZER,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PROC_COLUMN,
    RAY,
    SEQUENCE,
    SPLIT,
    TIED,
    TRAINER,
    TYPE,
)

from ludwig.schema.schema_utils import convert_submodules


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
