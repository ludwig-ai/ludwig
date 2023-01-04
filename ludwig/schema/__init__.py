from functools import lru_cache
from threading import Lock

from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    COMBINER,
    DEFAULTS,
    HYPEROPT,
    INPUT_FEATURES,
    LUDWIG_VERSION,
    MODEL_ECD,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    SPLIT,
    TRAINER,
)
from ludwig.schema.combiners.utils import get_combiner_jsonschema
from ludwig.schema.defaults.defaults import get_defaults_jsonschema
from ludwig.schema.features.utils import get_input_feature_jsonschema, get_output_feature_jsonschema
from ludwig.schema.hyperopt import get_hyperopt_jsonschema
from ludwig.schema.preprocessing import get_preprocessing_jsonschema
from ludwig.schema.trainer import get_model_type_jsonschema, get_trainer_jsonschema

VALIDATION_LOCK = Lock()


def get_ludwig_version_jsonschema():
    return {
        "type": "string",
        "title": "ludwig_version",
        "description": "Current Ludwig model schema version.",
    }


@DeveloperAPI
@lru_cache(maxsize=None)
def get_schema(model_type: str = MODEL_ECD):
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
        },
        "definitions": {},
        "required": [INPUT_FEATURES, OUTPUT_FEATURES],
        "additionalProperties": False,
    }

    if model_type == MODEL_ECD:
        schema["properties"][COMBINER] = get_combiner_jsonschema()

    return schema


@DeveloperAPI
@lru_cache(maxsize=1)
def get_validator():
    # Manually add support for tuples (pending upstream changes: https://github.com/Julian/jsonschema/issues/148):
    def custom_is_array(checker, instance):
        return isinstance(instance, list) or isinstance(instance, tuple)

    # This creates a new class, so cache to prevent a memory leak:
    # https://github.com/python-jsonschema/jsonschema/issues/868
    type_checker = Draft7Validator.TYPE_CHECKER.redefine("array", custom_is_array)
    return extend(Draft7Validator, type_checker=type_checker)


@DeveloperAPI
def validate_config(config):
    # Update config from previous versions to check that backwards compatibility will enable a valid config
    # NOTE: import here to prevent circular import
    from ludwig.data.split import get_splitter
    from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version

    # Update config from previous versions to check that backwards compatibility will enable a valid config
    updated_config = upgrade_config_dict_to_latest_version(config)
    model_type = updated_config.get(MODEL_TYPE, MODEL_ECD)

    splitter = get_splitter(**updated_config.get(PREPROCESSING, {}).get(SPLIT, {}))
    splitter.validate(updated_config)

    with VALIDATION_LOCK:
        validate(instance=updated_config, schema=get_schema(model_type=model_type), cls=get_validator())
