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
from threading import Lock

from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.constants import (
    COMBINER,
    DEFAULTS,
    HYPEROPT,
    INPUT_FEATURES,
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


@lru_cache(maxsize=2)
def get_schema(model_type: str = MODEL_ECD):
    schema = {
        "type": "object",
        "properties": {
            MODEL_TYPE: get_model_type_jsonschema(),
            INPUT_FEATURES: get_input_feature_jsonschema(),
            OUTPUT_FEATURES: get_output_feature_jsonschema(),
            COMBINER: get_combiner_jsonschema(),
            TRAINER: get_trainer_jsonschema(model_type),
            PREPROCESSING: get_preprocessing_jsonschema(),
            HYPEROPT: get_hyperopt_jsonschema(),
            DEFAULTS: get_defaults_jsonschema(),
        },
        "definitions": {},
        "required": [INPUT_FEATURES, OUTPUT_FEATURES],
    }
    return schema


@lru_cache(maxsize=2)
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
    from ludwig.data.split import get_splitter
    from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version

    # Update config from previous versions to check that backwards compatibility will enable a valid config
    updated_config = upgrade_config_dict_to_latest_version(config)
    model_type = updated_config.get(MODEL_TYPE, MODEL_ECD)

    splitter = get_splitter(**updated_config.get(PREPROCESSING, {}).get(SPLIT, {}))
    splitter.validate(updated_config)

    with VALIDATION_LOCK:
        # There is a race condition during schema validation that can cause the marshmallow schema class to
        # be missing during validation if more than one thread is trying to validate at once.
        validate(instance=updated_config, schema=get_schema(model_type=model_type), cls=get_validator())

    # Validate parameter interdependencies.
    # It may be possible that all of the following dynamic inter-parameter validations can be built into marshmallow
    # schemas. As these are built out, the following auxiliary validations can be gradually removed.
    # Check that all feature names are unique.

    # Check that all 'tied' parameters map to existing input feature names.

    # Check that 'dependent' features map to existing output features, and no circular dependencies.

    # Check that checkpoints_per_epoch and steps_per_checkpoint are both defined.

    # Check that validation_metric is actually a valid metric for one of the output features.

    # GBM model types don't work with the horovod backend.

    # If it's a ray backend feature[preprocessing][in_memory] must be true
    # def check_lazy_load_supported(self, feature):
    #     if not feature[PREPROCESSING]["in_memory"]:
    #         raise ValueError(
    #             f"RayBackend does not support lazy loading of data files at train time. "
    #             f"Set preprocessing config `in_memory: True` for feature {feature[NAME]}"
    #         )

    # If SequenceConcatCombiner, at least one of the input features should be a sequence feature.
    # if not seq_size:
    # raise ValueError("At least one of the input features for SequenceConcatCombiner should be a sequence.")

    # If TabTransformer, reduce_output is required.
    #  if config.reduce_output is None:
    # raise ValueError("TabTransformer requires the `reduce_output` " "parameter")

    # All of the feature names 'embed' for ComparatorCombiner are valid features.

    # Class balancing is only available for datasets with a single output feature.
    # if len(output_features) != 1:
    #     raise ValueError("Class balancing is only available for datasets with a single output feature")
    # if output_features[0][TYPE] != BINARY:
    #     raise ValueError("Class balancing is only supported for binary output types")

    # oversample minority and undersample majority are mutually exclusive.
    # if preprocessing_parameters["oversample_minority"] and preprocessing_parameters["undersample_majority"]:
    # raise ValueError(
    #     "Cannot balance data if both oversampling an undersampling are specified in the config. "
    #     "Must specify only one method"
    # )

    # raise ValueError(
    #     f"Filling missing values with mean is supported "
    #     f"only for number types, not for type {feature[TYPE]}.",
    # )

    # Audio input features, some must not be None.
    # if not getattr(self.encoder_obj.config, "embedding_size", None):
    #         raise ValueError("embedding_size has to be defined - " 'check "update_config_with_metadata()"')
    #     if not getattr(self.encoder_obj.config, "max_sequence_length", None):
    #         raise ValueError("max_sequence_length has to be defined - " 'check "update_config_with_metadata()"')

    # Check that all hyperopt feature names are real.

    # Check hyperopt output feature?
    # if output_feature == COMBINED:
    #     if metric != LOSS:
    #         raise ValueError('The only valid metric for "combined" output feature is "loss"')
    # else:
    #     output_feature_names = {of[NAME] for of in full_config[OUTPUT_FEATURES]}
    #     if output_feature not in output_feature_names:
    #         raise ValueError(
    #             'The output feature specified for hyperopt "{}" '
    #             "cannot be found in the config. "
    #             'Available ones are: {} and "combined"'.format(output_feature, output_feature_names)
    #         )

    # Check hyperopt metrics?
    # if metric not in feature_class.metric_functions:
    #         # todo v0.4: allow users to specify also metrics from the overall
    #         #  and per class metrics from the training stats and in general
    #         #  and post-processed metric
    #         raise ValueError(
    #             'The specified metric for hyperopt "{}" is not a valid metric '
    #             'for the specified output feature "{}" of type "{}". '
    #             "Available metrics are: {}".format(
    #                 metric, output_feature, output_feature_type, feature_class.metric_functions.keys()
    #             )
    #         )

    # GBMs only support 1 output feature
    # TODO: only single task currently
    # if len(output_feature_configs.to_dict()) > 1:
    #     raise ValueError("Only single task currently supported")

    # GBMs can't have non-tabular features.
