import copy
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List

import yaml
from marshmallow import ValidationError

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
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.modules.loss_modules import get_loss_cls
from ludwig.schema import validate_config
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.utils import combiner_registry
from ludwig.schema.decoders.utils import get_decoder_cls
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.encoders.base import PassthroughEncoderConfig
from ludwig.schema.encoders.binary_encoders import BinaryPassthroughEncoderConfig
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.utils import get_input_feature_cls, get_output_feature_cls, input_config_registry
from ludwig.schema.optimizers import get_optimizer_cls
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.split import get_split_cls
from ludwig.schema.trainer import BaseTrainerConfig, ECDTrainerConfig, GBMTrainerConfig
from ludwig.schema.utils import BaseMarshmallowConfig, convert_submodules
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.misc_utils import set_default_value

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


@dataclass(repr=False)
class ModelConfig(BaseMarshmallowConfig):
    """Configures the end-to-end LudwigModel machine learning pipeline.

    Refer to https://ludwig.ai/latest/configuration/ for full documentation.
    """

    def __init__(self, config_dict: dict):

        # ===== Backwards Compatibility =====
        upgraded_config_dict = self._upgrade_config(config_dict)

        # ===== Initialize Top Level Config Sections =====

        # Since 'ecd' is the default model type, we set it initially here.
        # If a user specifies a model type however, it will be overridden later.
        self.model_type: str = MODEL_ECD
        self.input_features: InputFeaturesContainer = InputFeaturesContainer()
        self.output_features: OutputFeaturesContainer = OutputFeaturesContainer()
        self.combiner: BaseCombinerConfig = ConcatCombinerConfig()
        self.trainer: BaseTrainerConfig = ECDTrainerConfig()
        self.preprocessing: PreprocessingConfig = PreprocessingConfig()
        self.defaults: DefaultsConfig = copy.deepcopy(DefaultsConfig())

        # ===== Set User Defined Global Defaults =====
        if DEFAULTS in upgraded_config_dict:
            self._set_attributes(self.defaults, upgraded_config_dict[DEFAULTS])

        # ===== Features =====
        self._set_feature_column(upgraded_config_dict)
        self._set_proc_column(upgraded_config_dict)
        self._initialize_input_features(upgraded_config_dict[INPUT_FEATURES])
        self._set_input_features(upgraded_config_dict[INPUT_FEATURES])
        self._initialize_output_features(upgraded_config_dict[OUTPUT_FEATURES])
        self._set_output_features(upgraded_config_dict[OUTPUT_FEATURES])

        # ===== Model Type =====
        if MODEL_TYPE in upgraded_config_dict:
            if upgraded_config_dict[MODEL_TYPE] == MODEL_GBM:
                self.model_type = MODEL_GBM
                self.trainer = GBMTrainerConfig()
                if (
                    TYPE in upgraded_config_dict.get(TRAINER, {})
                    and upgraded_config_dict[TRAINER][TYPE] != "lightgbm_trainer"
                ):
                    raise ValidationError("GBM Model trainer must be of type: 'lightgbm_trainer'")

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
        if COMBINER in upgraded_config_dict:
            if self.combiner.type != upgraded_config_dict[COMBINER][TYPE]:
                self.combiner = combiner_registry.get(upgraded_config_dict[COMBINER][TYPE]).get_schema_cls()()

            if self.combiner.type == SEQUENCE:
                encoder_family = SEQUENCE
            else:
                encoder_family = None
            self._set_attributes(self.combiner, upgraded_config_dict[COMBINER], feature_type=encoder_family)

        # ===== Trainer =====
        if TRAINER in upgraded_config_dict:
            self._set_attributes(self.trainer, upgraded_config_dict[TRAINER])

        # ===== Global Preprocessing =====
        if PREPROCESSING in upgraded_config_dict:
            self._set_attributes(self.preprocessing, upgraded_config_dict[PREPROCESSING])

        # ===== Hyperopt =====
        self.hyperopt = upgraded_config_dict.get(HYPEROPT, {})
        self._set_hyperopt_defaults()

        # ===== Validate Config =====
        self._validate_config(self.to_dict())

    def __repr__(self):
        config_repr = self.to_dict()
        config_repr[INPUT_FEATURES] = self.input_features.filter_features()
        config_repr[OUTPUT_FEATURES] = self.output_features.filter_features()
        config_repr[DEFAULTS] = self.defaults
        return yaml.dump(config_repr, sort_keys=False)

    @classmethod
    def from_dict(cls, dict_config):
        return cls(dict_config)

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path) as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Cannot parse input yaml file: {e}")
        return cls(yaml_config)

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
    def _upgrade_config(config_dict: dict) -> dict:
        """Helper function used to run backwards compatibility check on the config and return an upgraded version.

        Args:
            config_dict: Config Dictionary
        """
        return upgrade_config_dict_to_latest_version(config_dict)

    @staticmethod
    def _validate_config(config_dict: dict) -> None:
        """Helper function used to validate the config using the Ludwig Schema.

        Args:
            config_dict: Config Dictionary
        """
        validate_config(config_dict)

    @staticmethod
    def _get_config_nested_cls(section: str, section_type: str, feature_type: str) -> BaseMarshmallowConfig:
        """Helper function for getting the specified section class associated with the section type passed in.

        Args:
            section: Which nested config module we're dealing with.
            section_type: Which config schema to get (i.e. parallel_cnn)
            feature_type: feature type corresponding to config schema we're grabbing

        Returns:
            Config Schema to update the defaults section with.
        """
        if section == ENCODER:
            cls = get_encoder_cls(feature_type, section_type)
        elif section == DECODER:
            cls = get_decoder_cls(feature_type, section_type)
        elif section == LOSS:
            cls = get_loss_cls(feature_type, section_type).get_schema_cls()
        elif section == OPTIMIZER:
            cls = get_optimizer_cls(section_type)
        elif section == SPLIT:
            cls = get_split_cls(section_type)
        else:
            raise ValueError("Config Section not Supported")

        return copy.deepcopy(cls())

    def _initialize_input_features(self, feature_dicts: List[dict]) -> None:
        """This function initializes the input features on the ModelConfig that are specified in the user defined
        config dictionary. It does this by getting the corresponding feature config class, initializing it, then
        setting the encoder and preprocessing sections to the value of the corresponding global defaults section.
        By doing this, any global defaults that the user specified will be set here accordingly, however if no
        global defaults were specified, they will already reflect the regular defaults for each feature, so the
        input feature will initialize as expected. Any values set on the input feature itself will be set later in
        _set_input_features().

        Args:
            feature_dicts: List of input feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelConfig.
        """
        for feature_dict in feature_dicts:
            # Retrieve input feature schema cls from registry to initialize feature
            feature_config = copy.deepcopy(get_input_feature_cls(feature_dict[TYPE])())

            # Set global defaults on output feature config cls - if user has defined global defaults, these
            # will be set on the feature_config class, otherwise the global defaults already reflect the regular
            # defaults, so it will initialize the feature as expected.
            type_defaults = getattr(
                self.defaults, feature_dict[TYPE]
            )  # Global defaults section for specific feature type
            feature_config.encoder = copy.deepcopy(type_defaults.encoder)
            feature_config.preprocessing = copy.deepcopy(type_defaults.preprocessing)

            # Assign feature on output features container
            setattr(self.input_features, feature_dict[NAME], feature_config)

    def _set_input_features(self, feature_dicts: List[dict]) -> None:
        """This function sets the values on the ModelConfig that are specified on the input features themselves.
        This will override any global defaults that have been set in the previous function call
        _initialize_input_features().

        Args:
            feature_dicts: List of input feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelConfig.
        """
        for feature_dict in feature_dicts:
            # Set the parameters that the user specified on the input feature itself
            self._set_attributes(
                getattr(self.input_features, feature_dict[NAME]), feature_dict, feature_type=feature_dict[TYPE]
            )

    def _initialize_output_features(self, feature_dicts: List[dict]) -> None:
        """This function initializes the output features on the ModelConfig that are specified in the user defined
        config dictionary. It does this by getting the corresponding feature config class, initializing it, then
        setting the decoder and loss sections to the value of the corresponding global defaults section. By doing
        this, any global defaults that the user specified will be set here accordingly, however if no global
        defaults were specified, each section already reflects the regular defaults for each feature, so the output
        features will initialize as expected. Any values set on the output feature itself will be set later in
        _set_output_features().

        Args:
            feature_dicts: List of output feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelConfig.
        """
        for feature_dict in feature_dicts:
            # Retrieve output feature schema cls from registry to init feature
            feature_config = copy.deepcopy(get_output_feature_cls(feature_dict[TYPE])())

            # Set global defaults on output feature config cls - if user has defined global defaults, these
            # will be set on the feature_config class, otherwise the global defaults already reflect the regular
            # defaults, so it will initialize the feature as expected.
            type_defaults = getattr(
                self.defaults, feature_dict[TYPE]
            )  # Global defaults section for specific feature type
            feature_config.decoder = copy.deepcopy(type_defaults.decoder)
            feature_config.loss = copy.deepcopy(type_defaults.loss)

            # TODO depending on defaults section conversation may want to enable
            # feature_config.preprocessing = copy.deepcopy(type_defaults.preprocessing)

            # Assign feature on output features container
            setattr(self.output_features, feature_dict[NAME], feature_config)

    def _set_output_features(self, feature_dicts: List[dict]) -> None:
        """This function sets the values on the ModelConfig that are specified on the output features themselves.
        This will override any global defaults that have been set in the previous function call
        _initialize_output_features().

        Args:
            feature_dicts: List of output feature definitions in user defined config dict.

        Returns:
            None -> Updates ModelConfig.
        """
        for feature_dict in feature_dicts:
            # Set the parameters that the user specified on the output feature itself
            self._set_attributes(
                getattr(getattr(self, OUTPUT_FEATURES), feature_dict[NAME]),
                feature_dict,
                feature_type=feature_dict[TYPE],
            )

            # Set the reduce_input parameter for the tagger decoder specifically
            if getattr(getattr(self, OUTPUT_FEATURES), feature_dict[NAME]).decoder.type == "tagger":
                getattr(getattr(self, OUTPUT_FEATURES), feature_dict[NAME]).reduce_input = None

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

            #  Set new section for nested feature fields and recurse into function for setting values
            if key in [ENCODER, DECODER, LOSS, OPTIMIZER, SPLIT]:
                section = getattr(config_obj_lvl, key)

                # Check if nested section needs to be replaced
                if TYPE in val and section.type != val[TYPE]:
                    section = self._get_config_nested_cls(key, val[TYPE], feature_type)
                    setattr(config_obj_lvl, key, section)

                #  Now set the other defaults specified in the module
                self._set_attributes(section, val, feature_type=feature_type)

            # If val is a nested section (i.e. preprocessing) recurse into function to set values.
            elif isinstance(val, dict):
                self._set_attributes(getattr(config_obj_lvl, key), val, feature_type=feature_type)

            # Base case for setting values on leaves
            else:
                setattr(config_obj_lvl, key, val)

    def _set_gbm_attributes(self, config_dict: dict) -> None:
        """This function sets the appropriate attributes on the config object when the model type is 'gbm'. These
        are things such as the correct model trainer config class and passthrough encoders for the features.

        Args:
            config_dict: The user defined config dictionary.

        Returns:
            None -> Updates config object.
        """
        self.model_type = MODEL_GBM
        self.trainer = GBMTrainerConfig()
        if TYPE in config_dict.get(TRAINER, {}) and config_dict[TRAINER][TYPE] != "lightgbm_trainer":
            raise ValidationError("GBM Model trainer must be of type: 'lightgbm_trainer'")

        for feature in self.input_features.to_dict().keys():
            feature_cls = getattr(self.input_features, feature)
            if feature_cls.type == BINARY:
                feature_cls.encoder = BinaryPassthroughEncoderConfig()
            elif feature_cls.type in [CATEGORY, NUMBER]:
                feature_cls.encoder = PassthroughEncoderConfig()
            else:
                raise ValidationError("GBM Models currently only support Binary, Category, and Number " "features")

    def _set_hyperopt_defaults(self):
        """This function was migrated from defaults.py with the intention of setting some hyperopt defaults while
        the hyperopt section of the config object is not fully complete.

        Returns:
            None -> modifies trainer and hyperopt sections
        """
        if not self.hyperopt:
            return

        scheduler = self.hyperopt.get("executor", {}).get("scheduler")
        if not scheduler:
            return

        if EXECUTOR in self.hyperopt:
            set_default_value(self.hyperopt[EXECUTOR], TYPE, RAY)

        # Disable early stopping when using a scheduler. We achieve this by setting the parameter
        # to -1, which ensures the condition to apply early stopping is never met.
        early_stop = self.trainer.early_stop
        if early_stop is not None and early_stop != -1:
            warnings.warn("Can't utilize `early_stop` while using a hyperopt scheduler. Setting early stop to -1.")
        self.trainer.early_stop = -1

        max_t = scheduler.get("max_t")
        time_attr = scheduler.get("time_attr")
        epochs = self.trainer.to_dict().get("epochs", None)
        if max_t is not None:
            if time_attr == "time_total_s":
                if epochs is None:
                    setattr(self.trainer, "epochs", sys.maxsize)  # continue training until time limit hit
                # else continue training until either time or trainer epochs limit hit
            elif epochs is not None and epochs != max_t:
                raise ValueError(
                    "Cannot set trainer `epochs` when using hyperopt scheduler w/different training_iteration `max_t`. "
                    "Unset one of these parameters in your config or make sure their values match."
                )
            else:
                setattr(self.trainer, "epochs", max_t)  # run trainer until scheduler epochs limit hit
        elif epochs is not None:
            scheduler["max_t"] = epochs  # run scheduler until trainer epochs limit hit

    def update_with_dict(self, config_dict: dict):
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
        self._set_input_features(config_dict[INPUT_FEATURES])
        self._set_output_features(config_dict[OUTPUT_FEATURES])

        # ==== Combiner ====
        if COMBINER in config_dict:
            if self.combiner.type == SEQUENCE:

                # Encoder family will be used when getting sequence encoders for the sequence combiner. This is passed
                # into feature type so the sequence encoders set on this combiner will be available upon registry get
                encoder_family = SEQUENCE
            else:
                encoder_family = None

            # Set parameters on combiner with encoder family passed in
            self._set_attributes(self.combiner, config_dict[COMBINER], feature_type=encoder_family)

        # ==== Update Trainer ====
        if TRAINER in config_dict:
            self._set_attributes(self.trainer, config_dict[TRAINER])

    def to_dict(self) -> Dict[str, any]:
        """This method converts the current config object into an equivalent dictionary representation for the
        parts of the codebase that use the dictionary representation of the config.

        Returns:
            Config Dictionary
        """
        input_features = [feat for feat in self.input_features.to_list() if feat["active"]]
        output_features = [feat for feat in self.output_features.to_list() if feat["active"]]

        config_dict = {
            "model_type": self.model_type,
            "input_features": input_features,
            "output_features": output_features,
            "combiner": self.combiner.to_dict(),
            "trainer": self.trainer.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "hyperopt": self.hyperopt,
            "defaults": self.defaults.to_dict(),
        }
        return convert_submodules(config_dict)
