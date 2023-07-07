import copy
from abc import ABC
from typing import Any, Dict, Optional, Set

from marshmallow import ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.config_validation.checks import get_config_check_registry
from ludwig.config_validation.validation import check_schema
from ludwig.constants import (
    BACKEND,
    COLUMN,
    DEPENDENCIES,
    ENCODER,
    INPUT_FEATURES,
    MODEL_ECD,
    NAME,
    OUTPUT_FEATURES,
    TIED,
)
from ludwig.error import ConfigValidationError
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.base import BaseDefaultsConfig
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.utils import (
    merge_fixed_preprocessing_params,
    merge_with_defaults,
    sanitize_and_filter_combiner_entities_,
    set_derived_feature_columns_,
    set_hyperopt_defaults_,
    set_llm_tokenizers,
    set_preprocessing_parameters,
    set_tagger_decoder_parameters,
    set_validation_parameters,
)
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.schema.utils import ludwig_dataclass
from ludwig.types import ModelConfigDict
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.data_utils import get_sanitized_feature_name, load_yaml
from ludwig.utils.registry import Registry

model_type_schema_registry = Registry()


@DeveloperAPI
@ludwig_dataclass
class ModelConfig(schema_utils.BaseMarshmallowConfig, ABC):
    input_features: FeatureCollection[BaseInputFeatureConfig]
    output_features: FeatureCollection[BaseOutputFeatureConfig]

    model_type: str

    trainer: BaseTrainerConfig
    preprocessing: PreprocessingConfig
    defaults: BaseDefaultsConfig
    hyperopt: Optional[HyperoptConfig] = None

    backend: Dict[str, Any] = schema_utils.Dict()  # TODO(jeffkinnison): Add backend schema
    ludwig_version: str = schema_utils.ProtectedString(LUDWIG_VERSION)

    def __post_init__(self):
        merge_fixed_preprocessing_params(self)
        set_validation_parameters(self)
        set_hyperopt_defaults_(self)
        set_tagger_decoder_parameters(self)
        sanitize_and_filter_combiner_entities_(self)

        # Set preprocessing parameters for text features for LLM model type
        set_llm_tokenizers(self)

        # Reconcile conflicting preprocessing parameters
        set_preprocessing_parameters(self)

        # Derive proc_col for each feature from the feature's preprocessing parameters
        # after all preprocessing parameters have been set
        set_derived_feature_columns_(self)

        # Auxiliary checks.
        get_config_check_registry().check_config(self)

    @staticmethod
    def from_dict(config: ModelConfigDict) -> "ModelConfig":
        config = copy.deepcopy(config)
        config = upgrade_config_dict_to_latest_version(config)

        # Use sanitized feature names.
        # NOTE: This must be kept consistent with build_dataset()
        for input_feature in config[INPUT_FEATURES]:
            input_feature[NAME] = get_sanitized_feature_name(input_feature[NAME])
            if COLUMN in input_feature and input_feature[COLUMN]:
                input_feature[COLUMN] = get_sanitized_feature_name(input_feature[COLUMN])
        for output_feature in config[OUTPUT_FEATURES]:
            output_feature[NAME] = get_sanitized_feature_name(output_feature[NAME])
            if COLUMN in output_feature and output_feature[COLUMN]:
                output_feature[COLUMN] = get_sanitized_feature_name(output_feature[COLUMN])

        # Sanitize tied feature names.
        for input_feature in config[INPUT_FEATURES]:
            if TIED in input_feature and input_feature[TIED]:
                input_feature[TIED] = get_sanitized_feature_name(input_feature[TIED])

        # Sanitize dependent feature names.
        for output_feature in config[OUTPUT_FEATURES]:
            if DEPENDENCIES in output_feature and output_feature[DEPENDENCIES]:
                output_feature[DEPENDENCIES] = [
                    get_sanitized_feature_name(feature_name) for feature_name in output_feature[DEPENDENCIES]
                ]

        config["model_type"] = config.get("model_type", MODEL_ECD)
        model_type = config["model_type"]
        if model_type not in model_type_schema_registry:
            raise ValidationError(
                f"Invalid model type: '{model_type}', expected one of: {list(model_type_schema_registry.keys())}"
            )

        config = merge_with_defaults(config)

        # TODO(travis): handle this with helper function
        backend = config.get(BACKEND)
        if isinstance(backend, str):
            config[BACKEND] = {"type": backend}

        # JSON schema validation. Note that this is desireable on top of `schema.load(config)` below because marshmallow
        # deserialization permits additional properties while JSON schema validation, for schema (e.g. `trainer`) that
        # have `additionalProperties=False`, does not.
        #
        # Illustrative example: test_validate_config_misc.py::test_validate_no_trainer_type
        #
        # TODO: Set `additionalProperties=False` for all Ludwig schema, and look into passing in `unknown='RAISE'` to
        # marshmallow.load(), which raises an error for unknown fields during deserialization.
        # https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html#marshmallow.schema.Schema.load
        check_schema(config)

        cls = model_type_schema_registry[model_type]
        schema = cls.get_class_schema()()
        try:
            config_obj: ModelConfig = schema.load(config)
        except ValidationError as e:
            raise ConfigValidationError(f"Config validation error raised during config deserialization: {e}") from e

        return config_obj

    @staticmethod
    def from_yaml(config_path: str) -> "ModelConfig":
        return ModelConfig.from_dict(load_yaml(config_path))

    def get_feature_names(self) -> Set[str]:
        """Returns a set of all feature names."""
        feature_names = set()
        feature_names.update([f.column for f in self.input_features])
        feature_names.update([f.column for f in self.output_features])
        return feature_names

    def get_feature_config(self, feature_column_name: str) -> Optional[BaseInputFeatureConfig]:
        """Returns the feature config for the given feature name."""
        for feature in self.input_features:
            if feature.column == feature_column_name:
                return feature
        for feature in self.output_features:
            if feature.column == feature_column_name:
                return feature


@DeveloperAPI
def register_model_type(name: str):
    def wrap(model_type_config: ModelConfig) -> ModelConfig:
        model_type_schema_registry[name] = model_type_config
        return model_type_config

    return wrap


def _merge_encoder_cache_params(preprocessing_params: Dict[str, Any], encoder_params: Dict[str, Any]) -> Dict[str, Any]:
    if preprocessing_params.get("cache_encoder_embeddings"):
        preprocessing_params[ENCODER] = encoder_params
    return preprocessing_params
