from functools import lru_cache
from threading import Lock

from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.api_annotations import DeveloperAPI
from ludwig.config_validation import pre_checks
from ludwig.constants import (
    BACKEND,
    COMBINER,
    DEFAULTS,
    HYPEROPT,
    INPUT_FEATURES,
    LUDWIG_VERSION,
    MODEL_ECD,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TRAINER,
)
from ludwig.schema.combiners.utils import get_combiner_jsonschema
from ludwig.schema.defaults.defaults import get_defaults_jsonschema
from ludwig.schema.features.utils import get_input_feature_jsonschema, get_output_feature_jsonschema
from ludwig.schema.hyperopt import get_hyperopt_jsonschema
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.preprocessing import get_preprocessing_jsonschema
from ludwig.schema.trainer import get_model_type_jsonschema, get_trainer_jsonschema
from ludwig.types import ModelConfigDict

VALIDATION_LOCK = Lock()


def get_ludwig_version_jsonschema():
    return {
        "type": "string",
        "title": "ludwig_version",
        "description": "Current Ludwig model schema version.",
    }


def get_backend_jsonschema():
    # TODO(travis): implement full backend schema
    return {
        "type": "object",
        "title": "backend",
        "description": "Backend configuration.",
        "additionalProperties": True,
    }


@DeveloperAPI
@lru_cache(maxsize=2)
def get_schema(model_type: str = MODEL_ECD):
    # Force populate combiner registry:
    import ludwig.combiners.combiners  # noqa: F401

    schema = {
        "type": "object",
        "properties": {
            MODEL_TYPE: get_model_type_jsonschema(model_type),
            INPUT_FEATURES: get_input_feature_jsonschema(model_type),
            OUTPUT_FEATURES: get_output_feature_jsonschema(model_type),
            TRAINER: get_trainer_jsonschema(model_type),
            PREPROCESSING: get_preprocessing_jsonschema(),
            HYPEROPT: get_hyperopt_jsonschema(),
            DEFAULTS: get_defaults_jsonschema(),
            LUDWIG_VERSION: get_ludwig_version_jsonschema(),
            BACKEND: get_backend_jsonschema(),
        },
        "definitions": {},
        "required": [INPUT_FEATURES, OUTPUT_FEATURES],
        "additionalProperties": False,
    }

    if model_type == MODEL_ECD:
        schema["properties"][COMBINER] = get_combiner_jsonschema()

    return schema


@DeveloperAPI
@lru_cache(maxsize=2)
def get_validator():
    # Manually add support for tuples (pending upstream changes: https://github.com/Julian/jsonschema/issues/148):
    def custom_is_array(checker, instance):
        return isinstance(instance, list) or isinstance(instance, tuple)

    # This creates a new class, so cache to prevent a memory leak:
    # https://github.com/python-jsonschema/jsonschema/issues/868
    type_checker = Draft7Validator.TYPE_CHECKER.redefine("array", custom_is_array)
    return extend(Draft7Validator, type_checker=type_checker)


@DeveloperAPI
def validate_upgraded_config(updated_config):
    model_type = updated_config.get(MODEL_TYPE, MODEL_ECD)
    with VALIDATION_LOCK:
        validate(instance=updated_config, schema=get_schema(model_type=model_type), cls=get_validator())


@DeveloperAPI
def validate_config(config: ModelConfigDict):
    # Update config from previous versions to check that backwards compatibility will enable a valid config
    # NOTE: import here to prevent circular import
    from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version

    # Update config from previous versions to check that backwards compatibility will enable a valid config
    updated_config = upgrade_config_dict_to_latest_version(config)

    pre_checks.check_all_features_have_names_and_types(updated_config)

    # Schema validation.
    validate_upgraded_config(updated_config)

    comprehensive_config = ModelConfig(updated_config).to_dict()

    # Pre-checks.
    pre_checks.check_feature_names_unique(comprehensive_config)
    pre_checks.check_tied_features_are_valid(comprehensive_config)
    pre_checks.check_training_runway(comprehensive_config)
    pre_checks.check_dependent_features(comprehensive_config)
    pre_checks.check_gbm_horovod_incompatibility(comprehensive_config)
    pre_checks.check_gbm_single_output_feature(comprehensive_config)
    pre_checks.check_gbm_feature_types(comprehensive_config)
    pre_checks.check_gbm_trainer_type(comprehensive_config)
    pre_checks.check_ray_backend_in_memory_preprocessing(comprehensive_config)
    pre_checks.check_sequence_concat_combiner_requirements(comprehensive_config)
    pre_checks.check_comparator_combiner_requirements(comprehensive_config)
    pre_checks.check_class_balance_preprocessing(comprehensive_config)
    pre_checks.check_sampling_exclusivity(comprehensive_config)
    pre_checks.check_validation_metric_exists(comprehensive_config)
    pre_checks.check_splitter(comprehensive_config)
