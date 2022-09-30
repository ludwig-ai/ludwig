import copy
from typing import Dict, List

import yaml
from marshmallow import ValidationError

from ludwig.constants import (
    BINARY,
    CATEGORY,
    COLUMN,
    COMBINER,
    DECODER,
    DEFAULT_VALIDATION_METRIC,
    DEFAULTS,
    ENCODER,
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
    SEQUENCE,
    SPLIT,
    TIED,
    TRAINER,
    TYPE,
)
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.modules.loss_modules import get_loss_cls
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.utils import combiner_registry
from ludwig.schema.decoders.utils import get_decoder_cls
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.encoders.base import PassthroughEncoderConfig
from ludwig.schema.encoders.binary_encoders import BinaryPassthroughEncoderConfig
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.base import BaseFeatureConfig
from ludwig.schema.features.utils import get_input_feature_cls, get_output_feature_cls, input_config_registry
from ludwig.schema.optimizers import get_optimizer_cls
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.split import get_split_cls
from ludwig.schema.trainer import BaseTrainerConfig, ECDTrainerConfig, GBMTrainerConfig
from ludwig.schema.utils import BaseMarshmallowConfig, convert_submodules

DEFAULTS_MODULES = {NAME, COLUMN, PROC_COLUMN, TYPE, TIED, DEFAULT_VALIDATION_METRIC}


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


class InputFeaturesContainer(BaseFeatureContainer):
    """InputFeatures is a container for all input features."""

    pass


class OutputFeaturesContainer(BaseFeatureContainer):
    """OutputFeatures is a container for all output features."""

    pass


class Config:
    """This class is the implementation of the config object that replaces the need for a config dictionary
    throughout the project."""

    def __init__(self, config_dict: dict):

        self.model_type = MODEL_ECD
        self.input_features: InputFeaturesContainer = InputFeaturesContainer()
        self.output_features: OutputFeaturesContainer = OutputFeaturesContainer()
        self.combiner: BaseCombinerConfig = copy.deepcopy(ConcatCombinerConfig())
        self.trainer: BaseTrainerConfig = copy.deepcopy(ECDTrainerConfig())
        self.preprocessing: PreprocessingConfig = copy.deepcopy(PreprocessingConfig())
        self.hyperopt = config_dict.get(HYPEROPT, {})
        self.defaults: DefaultsConfig = copy.deepcopy(DefaultsConfig())

        # ===== Defaults =====
        if DEFAULTS in config_dict:
            self._set_attributes(self.defaults, config_dict[DEFAULTS])

        # ===== Features =====
        self._set_feature_column(config_dict)
        self._set_proc_column(config_dict)
        self._parse_features(config_dict[INPUT_FEATURES], INPUT_FEATURES)
        self._parse_features(config_dict[OUTPUT_FEATURES], OUTPUT_FEATURES)

        # ===== Model Type =====
        if MODEL_TYPE in config_dict:
            if config_dict[MODEL_TYPE] == MODEL_GBM:
                self.model_type = MODEL_GBM
                self.trainer = GBMTrainerConfig()
                if TYPE in config_dict.get(TRAINER, {}):
                    assert config_dict[TRAINER][TYPE] == "lightgbm_trainer"

                for feature in self.input_features.to_dict().keys():
                    feature_cls = getattr(self.input_features, feature)
                    if feature_cls.type == BINARY:
                        feature_cls.encoder = BinaryPassthroughEncoderConfig()
                    elif feature_cls.type in [CATEGORY, NUMBER]:
                        feature_cls.encoder = PassthroughEncoderConfig()
                    else:
                        raise ValidationError(
                            "GBM Models currently only support Binary, Category, and Number " "features"
                        )

        # ===== Combiner =====
        if COMBINER in config_dict:
            if self.combiner.type != config_dict[COMBINER][TYPE]:
                self.combiner = combiner_registry.get(config_dict[COMBINER][TYPE]).get_schema_cls()()

            if self.combiner.type == SEQUENCE:
                encoder_family = SEQUENCE
            else:
                encoder_family = None
            self._set_attributes(self.combiner, config_dict[COMBINER], feature_type=encoder_family)

        # ===== Trainer =====
        if TRAINER in config_dict:
            self._set_attributes(self.trainer, config_dict[TRAINER])

        # ===== Global Preprocessing =====
        if PREPROCESSING in config_dict:
            self._set_attributes(self.preprocessing, config_dict[PREPROCESSING])

        # ===== Hyperopt =====
        if HYPEROPT in config_dict:
            pass  # TODO: Schemify Hyperopt

    def __repr__(self):
        output = ""
        output += f"Model Type: {self.model_type}\n"
        output += f"Input Features: {yaml.dump()}"
        return output

    @staticmethod
    def _set_feature_column(config: dict) -> None:
        for feature in config[INPUT_FEATURES] + config[OUTPUT_FEATURES]:
            if COLUMN not in feature:
                feature[COLUMN] = feature[NAME]

    @staticmethod
    def _set_proc_column(config: dict) -> None:
        for feature in config[INPUT_FEATURES] + config[OUTPUT_FEATURES]:
            if PROC_COLUMN not in feature:
                feature[PROC_COLUMN] = compute_feature_hash(feature)

    @staticmethod
    def _get_new_config(module: str, config_type: str, feature_type: str) -> BaseMarshmallowConfig:
        """Helper function for getting the appropriate config to set in defaults section.

        Args:
            module: Which nested config module we're dealing with.
            config_type: Which config schema to get (i.e. parallel_cnn)
            feature_type: feature type corresponding to config schema we're grabbing

        Returns:
            Config Schema to update the defaults section with.
        """
        if module == ENCODER:
            return get_encoder_cls(feature_type, config_type)

        if module == DECODER:
            return get_decoder_cls(feature_type, config_type)

        if module == LOSS:
            return get_loss_cls(feature_type, config_type).get_schema_cls()

        if module == OPTIMIZER:
            return get_optimizer_cls(config_type)

        if module == SPLIT:
            return get_split_cls(config_type)

        raise ValueError("Module needs to be added to parsing support")

    def _parse_features(self, features: List[dict], feature_section: str, initialize: bool = True):
        """This function sets the values on the config object that are specified in the user defined config
        dictionary.

        Note: Sometimes features in tests have both an encoder and decoder specified. This causes issues in the config
              obj, so we make sure to check and remove inappropriate modules.
        Args:
            features: List of feature definitions in user defined config dict.
            feature_section: Indication of input features vs. output features.

        Returns:
            None -> Updates config object.
        """
        for feature in features:
            if feature_section == INPUT_FEATURES:
                if DECODER in feature:  # Ensure input feature doesn't have decoder specs
                    del feature[DECODER]
                feature_config = get_input_feature_cls(feature[TYPE])()  # name something else
                if initialize:
                    updated_feature_config = self._update_with_global_defaults(
                        feature_config, feature[TYPE], feature_section
                    )
                    setattr(self.input_features, feature[NAME], updated_feature_config)
                self._set_attributes(getattr(self.input_features, feature[NAME]), feature, feature_type=feature[TYPE])

            else:
                if ENCODER in feature:  # Ensure output feature doesn't have encoder specs
                    del feature[ENCODER]
                feature_config = get_output_feature_cls(feature[TYPE])()
                if initialize:
                    updated_feature_config = self._update_with_global_defaults(
                        feature_config, feature[TYPE], feature_section
                    )
                    setattr(self.output_features, feature[NAME], updated_feature_config)
                self._set_attributes(
                    getattr(getattr(self, feature_section), feature[NAME]), feature, feature_type=feature[TYPE]
                )
                if getattr(getattr(self, feature_section), feature[NAME]).decoder.type == "tagger":
                    getattr(getattr(self, feature_section), feature[NAME]).reduce_input = None

    def _set_attributes(self, config_obj_lvl: BaseMarshmallowConfig, config_dict_lvl: dict, feature_type: str = None):
        """
        This function recursively parses both config object from the point that's passed in and the config dictionary to
        make sure the config obj section in question matches the corresponding user specified config section.
        Args:
            config_obj_lvl: The level of the config object we're currently at.
            config_dict_lvl: The level of the config dict we're currently at.
            feature_type: The feature type to be piped into recursive calls for registry retrievals.

        Returns:
            None -> Updates config object.
        """
        for key, val in config_dict_lvl.items():

            # Persist feature type for getting schemas from registries
            if key in input_config_registry.keys():
                feature_type = key

            #  Update logic for nested feature fields
            if key in [ENCODER, DECODER, LOSS, OPTIMIZER, SPLIT]:
                module = getattr(config_obj_lvl, key)

                # Check if submodule needs update
                if TYPE in val and module.type != val[TYPE]:
                    new_config = self._get_new_config(key, val[TYPE], feature_type)()
                    setattr(config_obj_lvl, key, new_config)

                #  Now set the other defaults specified in the module
                self._set_attributes(getattr(config_obj_lvl, key), val, feature_type=feature_type)

            elif isinstance(val, dict):
                self._set_attributes(getattr(config_obj_lvl, key), val, feature_type=feature_type)

            else:
                setattr(config_obj_lvl, key, val)

    def _update_with_global_defaults(
        self, feature: BaseFeatureConfig, feat_type: str, feature_section: str
    ) -> BaseFeatureConfig:
        """This purpose of this function is to set the attributes of the features that are specified in the
        defaults section of the config.

        Args:
            feature: The feature with attributes to be set from specified defaults.
            feat_type: The feature type use to get the defaults to use for parameter setting.
            feature_section: Input features or Output features switch

        Returns:
            The feature with defaults set.
        """
        type_defaults = getattr(self.defaults, feat_type)
        config_sections = feature.to_dict().keys()

        for section in config_sections:
            if feature_section == INPUT_FEATURES:
                if section in {ENCODER, PREPROCESSING}:
                    setattr(feature, section, copy.deepcopy(getattr(type_defaults, section)))
            else:
                if section in {DECODER, LOSS}:
                    setattr(feature, section, copy.deepcopy(getattr(type_defaults, section)))

        return feature

    def update_config_object(self, config_dict: dict):
        """This function enables the functionality to update the config object with the config dict in case it has
        been altered by a particular section of the Ludwig pipeline. For example, preprocessing/auto_tune_config
        make changes to the config dict that need to be reconciled with the config obj. This function will ideally
        be removed once the entire codebase conforms to the config object, but until then, it will be very helpful!

        Args:
            config_dict: Altered config dict to use when reconciling changes

        Returns:
            None -> Alters config object
        """
        # ==== Update Features ====
        self._parse_features(config_dict[INPUT_FEATURES], INPUT_FEATURES, initialize=False)
        self._parse_features(config_dict[OUTPUT_FEATURES], OUTPUT_FEATURES, initialize=False)

        # ==== Combiner ====
        if COMBINER in config_dict:
            if self.combiner.type == SEQUENCE:
                encoder_family = SEQUENCE
            else:
                encoder_family = None
            self._set_attributes(self.combiner, config_dict[COMBINER], feature_type=encoder_family)

        # ==== Update Trainer ====
        if TRAINER in config_dict:
            self._set_attributes(self.trainer, config_dict[TRAINER])

    def get_config_dict(self) -> Dict[str, any]:
        """This method converts the current config object into an equivalent dictionary representation since many
        parts of the codebase still use the dictionary representation of the config.

        Returns:
            Config Dictionary
        """
        config_dict = {
            "model_type": self.model_type,
            "input_features": self.input_features.to_list(),
            "output_features": self.output_features.to_list(),
            "combiner": self.combiner.to_dict(),
            "trainer": self.trainer.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "hyperopt": self.hyperopt,
            "defaults": self.defaults.to_dict(),
        }
        return convert_submodules(config_dict)
