from functools import lru_cache
from threading import Lock

import jsonschema.exceptions
from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BASE_MODEL, MODEL_ECD, MODEL_LLM, MODEL_TYPE
from ludwig.error import ConfigValidationError

# TODO(travis): figure out why we need these imports to avoid circular import error
from ludwig.schema.combiners.utils import get_combiner_jsonschema  # noqa
from ludwig.schema.features.utils import get_input_feature_jsonschema, get_output_feature_jsonschema  # noqa
from ludwig.schema.hyperopt import get_hyperopt_jsonschema  # noqa
from ludwig.schema.trainer import get_model_type_jsonschema, get_trainer_jsonschema  # noqa
from ludwig.schema.utils import unload_jsonschema_from_marshmallow_class

VALIDATION_LOCK = Lock()


@DeveloperAPI
@lru_cache(maxsize=3)
def get_schema(model_type: str = MODEL_ECD):
    # Force populate combiner registry:
    import ludwig.combiners.combiners  # noqa: F401
    from ludwig.schema.model_types.base import model_type_schema_registry

    cls = model_type_schema_registry[model_type]
    props = unload_jsonschema_from_marshmallow_class(cls)["properties"]

    # TODO: Replace with more robust required logic later.
    required = ["input_features", "output_features"]
    if model_type == MODEL_LLM:
        required += [BASE_MODEL]

    return {
        "type": "object",
        "properties": props,
        "title": "model_options",
        "description": "Settings for Ludwig configuration",
        "required": required,
        "additionalProperties": True,  # TODO: Set to false after 0.8 releases.
    }


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
def check_schema(updated_config):
    """Emulates the pure JSONSchema validation that could be used in an environment without marshmallow.

    The incoming config may not be comprehensive, but is assumed to be up to date with the latest ludwig schema.
    """
    model_type = updated_config.get(MODEL_TYPE, MODEL_ECD)
    error = None
    with VALIDATION_LOCK:
        try:
            validate(instance=updated_config, schema=get_schema(model_type=model_type), cls=get_validator())
        except jsonschema.exceptions.ValidationError as e:
            # Capture error but don't raise here, otherwise we get the full output from `e`, which contains a dump
            # of the entire schema
            error = e

    if error is not None:
        raise ConfigValidationError(f"Failed to validate JSON schema for config. Error: {error.message}") from error
