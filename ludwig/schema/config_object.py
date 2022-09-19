from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.trainer import BaseTrainerConfig, ECDTrainerConfig, GBMTrainerConfig
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.utils import input_type_registry, output_type_registry
from ludwig.schema.combiners.utils import combiner_registry

from ludwig.constants import (
    COMBINER,
    DECODER,
    MODEL_ECD,
    ENCODER,
    MODEL_GBM,
    HYPEROPT,
    INPUT_FEATURES,
    MODEL_TYPE,
    NAME,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TRAINER,
    TYPE,
)


class InputFeatures:
    """
    InputFeatures is a container for all input features.
    """

    def to_dict(self):
        return self.__dict__


class OutputFeatures:
    """
    OutputFeatures is a container for all output features.
    """

    def to_dict(self):
        return self.__dict__


class Config:
    """
    This class is the implementation of the config object that replaces the need for a config dictionary throughout the
    project.
    """
    model_type = MODEL_ECD
    input_features = InputFeatures()
    output_features = OutputFeatures()
    combiner: BaseCombinerConfig = ConcatCombinerConfig()
    trainer: BaseTrainerConfig = ECDTrainerConfig()
    preprocessing = PreprocessingConfig()
    hyperopt = {}
    defaults = DefaultsConfig()

    def __init__(self, config_dict):
        self.parse_input_features(config_dict[INPUT_FEATURES])
        self.parse_output_features(config_dict[OUTPUT_FEATURES])

        if MODEL_TYPE in config_dict:
            if config_dict[MODEL_TYPE] == MODEL_GBM:
                self.model_type = MODEL_GBM
                self.trainer = GBMTrainerConfig()

        if COMBINER in config_dict:
            self.combiner = combiner_registry.get(config_dict[COMBINER][TYPE])
            self.set_attributes(self.combiner, config_dict[COMBINER])

        if TRAINER in config_dict:
            self.set_attributes(self.trainer, config_dict[TRAINER])

        if PREPROCESSING in config_dict:
            self.set_attributes(self.preprocessing, config_dict[PREPROCESSING])

        if HYPEROPT in config_dict:
            pass
            # self.set_attributes(self.hyperopt, config_dict[HYPEROPT])

    def parse_input_features(self, input_features):
        for feature in input_features:
            if DECODER in feature:
                del feature[DECODER]
            feature_schema = input_type_registry[feature[TYPE]].get_schema_cls()
            setattr(self.input_features, feature[NAME], feature_schema())
            self.set_attributes(getattr(self.input_features, feature[NAME]), feature)

    def parse_output_features(self, output_features):
        for feature in output_features:
            if ENCODER in feature:
                del feature[ENCODER]
            feature_schema = output_type_registry[feature[TYPE]].get_schema_cls()
            setattr(self.output_features, feature[NAME], feature_schema())
            self.set_attributes(getattr(self.output_features, feature[NAME]), feature)

    def set_attributes(self, attribute, value):
        for key, val in value.items():
            if isinstance(val, dict):
                self.set_attributes(getattr(attribute, key), val)
            else:
                setattr(attribute, key, val)