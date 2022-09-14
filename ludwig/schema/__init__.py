#
# Module structure:
# ludwig.schema               <-- Meant to contain all schemas, utilities, helpers related to describing and validating
#                                 Ludwig configs.
# ├── __init__.py             <-- Contains fully assembled Ludwig schema (`get_schema()`), `validate_config()` for YAML
#                                 validation, and all "top-level" schema functions.
# ├── utils.py                <-- An extensive set of marshmallow-related fields, methods, and schemas that are used
#                                 elsewhere in Ludwig.
# ├── trainer.py              <-- Contains `TrainerConfig()` and `get_trainer_jsonschema`
# ├── optimizers.py           <-- Contains every optimizer config (e.g. `SGDOptimizerConfig`, `AdamOptimizerConfig`,
#                                 etc.) and related marshmallow fields/methods.
# └── combiners/
#     ├── __init__.py         <-- Imports for each combiner config file (making imports elsewhere more convenient).
#     ├── utils.py            <-- Location of `combiner_registry`, `get_combiner_jsonschema()`, `get_combiner_conds()`
#     ├── base.py             <-- Location of `BaseCombinerConfig`
#     ├── comparator.py       <-- Location of `ComparatorCombinerConfig`
#     ... <file for each combiner> ...
#     └──  transformer.py     <-- Location of `TransformerCombinerConfig`
#

from functools import lru_cache

from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.constants import (
    COMBINER,
    DEFAULTS,
    HYPEROPT,
    INPUT_FEATURES,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TRAINER,
)
from ludwig.schema.combiners.utils import get_combiner_jsonschema
from ludwig.schema.defaults.defaults import get_defaults_jsonschema
from ludwig.schema.features.utils import get_input_feature_jsonschema, get_output_feature_jsonschema
from ludwig.schema.preprocessing import get_preprocessing_jsonschema
from ludwig.schema.trainer import get_model_type_jsonschema, get_trainer_jsonschema


@lru_cache(maxsize=1)
def get_schema():
    schema = {
        "type": "object",
        "properties": {
            MODEL_TYPE: get_model_type_jsonschema(),
            INPUT_FEATURES: get_input_feature_jsonschema(),
            OUTPUT_FEATURES: get_output_feature_jsonschema(),
            COMBINER: get_combiner_jsonschema(),
            TRAINER: get_trainer_jsonschema(),
            PREPROCESSING: get_preprocessing_jsonschema(),
            HYPEROPT: {},
            DEFAULTS: get_defaults_jsonschema(),
        },
        "definitions": {},
        "required": [INPUT_FEATURES, OUTPUT_FEATURES],
    }
    return schema


@lru_cache(maxsize=1)
def get_validator():
    # Manually add support for tuples (pending upstream changes: https://github.com/Julian/jsonschema/issues/148):
    def custom_is_array(checker, instance):
        return isinstance(instance, list) or isinstance(instance, tuple)

    # This creates a new class, so cache to prevent a memory leak:
    # https://github.com/python-jsonschema/jsonschema/issues/868
    type_checker = Draft7Validator.TYPE_CHECKER.redefine("array", custom_is_array)
    return extend(Draft7Validator, type_checker=type_checker)


def validate_config(config):
    # Update config from previous versions to check that backwards compatibility will enable a valid config
    # NOTE: import here to prevent circular import
    from ludwig.utils.backward_compatibility import upgrade_to_latest_version

    # Update config from previous versions to check that backwards compatibility will enable a valid config
    updated_config = upgrade_to_latest_version(config)

    validate(instance=updated_config, schema=get_schema(), cls=get_validator())
