from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.trainer import BaseTrainerConfig, ECDTrainerConfig
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.features.utils import input_type_registry, output_type_registry

from ludwig.constants import (
    COMBINER,
    HYPEROPT,
    INPUT_FEATURES,
    NAME,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TRAINER,
    TYPE,
)


class Config:
    """
    This class is the implementation of the config object that replaces the need for a config dictionary throughout the
    project.
    """
    input_features: list = []
    combiner: BaseCombinerConfig = ConcatCombinerConfig()
    output_features: list = []
    trainer: BaseTrainerConfig = ECDTrainerConfig()
    preprocessing = PreprocessingConfig()
    hyperopt = {}

    def __init__(self, config_dict):
        self.parse_input_features(config_dict[INPUT_FEATURES])

        if COMBINER in config_dict:
            self.parse_combiner(config_dict[COMBINER])

        self.parse_output_features(config_dict[OUTPUT_FEATURES])

        if TRAINER in config_dict:
            self.parse_trainer(config_dict[TRAINER])

        if PREPROCESSING in config_dict:
            self.parse_preprocessing(config_dict[PREPROCESSING])

        if HYPEROPT in config_dict:
            self.parse_hyperopt(config_dict[HYPEROPT])

    def parse_input_features(self, input_features):
        for feature in input_features:
            feature_schema = input_type_registry[feature[TYPE]].get_schema_cls()
            setattr(self.input_features, feature[NAME], feature_schema())

    def parse_output_features(self, output_features):
        for feature in output_features:
            feature_schema = output_type_registry[feature[TYPE]].get_schema_cls()
            setattr(self.output_features, feature[NAME], feature_schema())

    def parse_combiner(self, combiner):
        for key, value in combiner.items():
            setattr(self.combiner, key, value)

    def parse_trainer(self, trainer):
        for key, value in trainer.items():
            setattr(self.trainer, key, value)

    def parse_preprocessing(self, preprocessing):
        for key, value in preprocessing.items():
            setattr(self.preprocessing, key, value)

    def parse_hyperopt(self, hyperopt):
        pass


