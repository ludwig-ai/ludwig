#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
import logging
import warnings
from typing import Any, Callable, Dict, List, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BIAS,
    CLASS_WEIGHTS,
    COLUMN,
    CONV_BIAS,
    CONV_USE_BIAS,
    DECODER,
    DEFAULT_BIAS,
    DEFAULT_USE_BIAS,
    DEFAULTS,
    ENCODER,
    EVAL_BATCH_SIZE,
    EXECUTOR,
    FORCE_SPLIT,
    HEIGHT,
    HYPEROPT,
    IMAGE,
    INPUT_FEATURES,
    LOSS,
    MISSING_VALUE_STRATEGY,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NAME,
    NUM_SAMPLES,
    NUMBER,
    OUTPUT_FEATURES,
    PARAMETERS,
    PREPROCESSING,
    PROBABILITIES,
    RANDOM,
    RAY,
    SAMPLER,
    SCHEDULER,
    SEARCH_ALG,
    SEQUENCE,
    SPLIT,
    SPLIT_PROBABILITIES,
    STRATIFY,
    TEXT,
    TIMESERIES,
    TRAINER,
    TRAINING,
    TYPE,
    USE_BIAS,
    WIDTH,
)
from ludwig.features.feature_registries import get_base_type_registry, get_input_type_registry, get_output_type_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema.defaults.gbm import GBMDefaultsConfig
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.types import (
    FeatureConfigDict,
    FeatureTypeDefaultsDict,
    HyperoptConfigDict,
    ModelConfigDict,
    PreprocessingConfigDict,
    TrainerConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.misc_utils import get_from_registry, merge_dict
from ludwig.utils.version_transformation import VersionTransformation, VersionTransformationRegistry

config_transformation_registry = VersionTransformationRegistry()


@DeveloperAPI
def register_config_transformation(version: str, prefixes: Union[str, List[str]] = []) -> Callable:
    """This decorator registers a transformation function for a config version. Version is the first version which
    requires the transform. For example, since "training" is renamed to "trainer" in 0.5, this change should be
    registered with 0.5.  from_version < version <= to_version.

    Args:
        version: The version to register this transformation with. The earliest ludwig version which requires this
                 transformation.
        prefixes: A list of keypath prefixes to apply this transformation to. If not specified, transforms the entire
                  config dict. If a prefix indicates a list, i.e. "input_features", the transformation is applied to
                  each element of the list (each input feature).
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]

    def wrap(fn: Callable[[Dict], Dict]):
        config_transformation_registry.register(VersionTransformation(transform=fn, version=version, prefixes=prefixes))
        return fn

    return wrap


@DeveloperAPI
def upgrade_config_dict_to_latest_version(config: ModelConfigDict) -> ModelConfigDict:
    """Updates config from an older version of Ludwig to the current version. If config does not have a
    "ludwig_version" key, all updates are applied.

    Args:
        config: A config saved by an older version of Ludwig.

    Returns A new copy of config, upgraded to the current Ludwig version. Returns config if config has no
            "ludwig_version".
    """
    return config_transformation_registry.update_config(
        config, from_version=config.get("ludwig_version", "0.0"), to_version=LUDWIG_VERSION
    )


def upgrade_model_progress(model_progress: Dict) -> Dict:
    """Updates model progress info to be compatible with latest ProgressTracker implementation.

    Notably, we convert epoch-based stats to their step-based equivalents and reformat metrics into `TrainerMetric`
    tuples.
    """
    ret = copy.deepcopy(model_progress)

    if "last_improvement_epoch" in ret:
        ret["last_improvement_steps"] = ret["last_improvement_epoch"] * ret["batch_size"]
        del ret["last_improvement_epoch"]

    if "last_learning_rate_reduction_epoch" in ret:
        ret["last_learning_rate_reduction_steps"] = ret["last_learning_rate_reduction_epoch"] * ret["batch_size"]
        del ret["last_learning_rate_reduction_epoch"]

    if "last_increase_batch_size_epoch" in ret:
        ret["last_increase_batch_size_steps"] = ret["last_increase_batch_size_epoch"] * ret["batch_size"]
        del ret["last_increase_batch_size_epoch"]

    if "vali_metrics" in ret:
        ret["validation_metrics"] = ret["vali_metrics"]
        del ret["vali_metrics"]

    for metric_group in ("train_metrics", "test_metrics", "validation_metrics"):
        if metric_group not in ret:
            continue
        for tgt in ret[metric_group]:
            for metric in ret[metric_group][tgt]:
                if len(ret[metric_group][tgt][metric]) == 0 or isinstance(
                    ret[metric_group][tgt][metric][0], (tuple, list)
                ):
                    continue

                ret[metric_group][tgt][metric] = [
                    TrainerMetric(i + 1, (i + 1) * ret["batch_size"], val)
                    for i, val in enumerate(ret[metric_group][tgt][metric])
                ]

    if "tune_checkpoint_num" not in ret:
        ret["tune_checkpoint_num"] = 0

    # Upgrades related to extending progress tracker with explicit bests.
    if "checkpoint_number" not in ret:
        ret["checkpoint_number"] = 0

    if "best_eval_metric_steps" not in ret:
        ret["best_eval_metric_steps"] = 0

    if "best_eval_metric_epoch" not in ret:
        ret["best_eval_metric_epoch"] = 0

    if "best_eval_metric_checkpoint_number" not in ret:
        ret["best_eval_metric_checkpoint_number"] = 0

    if "best_eval_train_metrics" not in ret:
        ret["best_eval_train_metrics"] = {}

    if "best_eval_validation_metrics" not in ret:
        ret["best_eval_validation_metrics"] = {}

    if "best_eval_test_metrics" not in ret:
        ret["best_eval_test_metrics"] = {}

    if "best_eval_metric" in ret:
        ret["best_eval_metric_value"] = ret["best_eval_metric"]
        del ret["best_eval_metric"]

    if "last_improvement" in ret:
        del ret["last_improvement"]

    # Delete learning-rate related fields removed in https://github.com/ludwig-ai/ludwig/pull/2877.
    if "best_reduce_learning_rate_eval_metric" in ret:
        del ret["best_reduce_learning_rate_eval_metric"]

    if "last_reduce_learning_rate_eval_metric_improvement" in ret:
        del ret["last_reduce_learning_rate_eval_metric_improvement"]

    return ret


def _traverse_dicts(config: Any, f: Callable[[Dict], None]):
    """Recursively applies function f to every dictionary contained in config.

    f should in-place modify the config dict. f will be called on leaves first, root last.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            _traverse_dicts(v, f)
        f(config)
    elif isinstance(config, list):
        for v in config:
            _traverse_dicts(v, f)


@register_config_transformation("0.6", "backend")
def _update_backend_cache_credentials(backend: Dict[str, Any]) -> Dict[str, Any]:
    if "cache_credentials" in backend:
        credentials = backend.get("credentials", {})
        if "cache" in credentials:
            warnings.warn("`cache` already found in `backend.credentials`, ignoring `cache_credentials`")
        else:
            warnings.warn(
                "`backend.cache_credentials` has been renamed `backend.credentials.cache`", DeprecationWarning
            )
            credentials["cache"] = backend.pop("cache_credentials")
        backend["credentials"] = credentials
    return backend


@register_config_transformation("0.6", ["output_features"])
def update_class_weights_in_features(feature: FeatureConfigDict) -> FeatureConfigDict:
    if LOSS in feature:
        class_weights = feature[LOSS].get(CLASS_WEIGHTS, None)
        if not isinstance(class_weights, (list, dict)):
            class_weights = None
        feature[LOSS][CLASS_WEIGHTS] = class_weights

    return feature


@register_config_transformation("0.4")
def _update_level_metadata(config: ModelConfigDict) -> ModelConfigDict:
    # Replace parameters represented as keys with params represented as values.
    # Precedence is defined by first in the dictionary order, so if multiple
    # provided keys map to the same value, the one that appears earlier in this
    # dictionary will take priority.
    drop_params = {
        "sequence_length_limit": "max_sequence_length",
        "word_most_common": "most_common",
        "word_sequence_length_limit": "max_sequence_length",
        "word_tokenizer": "tokenizer",
        "word_vocab_file": "vocab_file",
        "char_most_common": "most_common",
        "char_sequence_length_limit": "max_sequence_length",
        "char_tokenizer": "tokenizer",
        "char_vocab_file": "vocab_file",
    }

    def upgrade_params(params):
        for key, value in drop_params.items():
            if key in params:
                if value in params:
                    warnings.warn(
                        f"Removing deprecated config preprocessing parameter {key} as new param {value} already "
                        f"present in the config",
                        DeprecationWarning,
                    )
                else:
                    warnings.warn(
                        f"Renaming deprecated config preprocessing parameter {key} to {value}",
                        DeprecationWarning,
                    )
                    params[value] = params[key]
                del params[key]

    sequence_types = [SEQUENCE, TEXT, AUDIO, TIMESERIES]
    for dtype in sequence_types:
        params = config.get(PREPROCESSING, {}).get(dtype, {})
        upgrade_params(params)

    for feature in config[INPUT_FEATURES]:
        if feature.get(TYPE) not in sequence_types:
            continue
        params = feature.get(PREPROCESSING, {})
        upgrade_params(params)

    return config


@register_config_transformation("0.5")
def rename_training_to_trainer(config: ModelConfigDict) -> ModelConfigDict:
    if TRAINING in config:
        warnings.warn('Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning)
        config[TRAINER] = config[TRAINING]
        del config[TRAINING]
    return config


@register_config_transformation("0.5", ["input_features", "output_features"])
def _upgrade_use_bias_in_features(feature: FeatureConfigDict) -> FeatureConfigDict:
    def upgrade_use_bias(config):
        if BIAS in config:
            warnings.warn('Parameter "bias" renamed to "use_bias" and will be removed in v0.6', DeprecationWarning)
            config[USE_BIAS] = config[BIAS]
            del config[BIAS]
        if CONV_BIAS in config:
            warnings.warn(
                'Parameter "conv_bias" renamed to "conv_use_bias" and will be removed in v0.6', DeprecationWarning
            )
            config[CONV_USE_BIAS] = config[CONV_BIAS]
            del config[CONV_BIAS]
        if DEFAULT_BIAS in config:
            warnings.warn(
                'Parameter "default_bias" renamed to "default_use_bias" and will be removed in v0.6', DeprecationWarning
            )
            config[DEFAULT_USE_BIAS] = config[DEFAULT_BIAS]
            del config[DEFAULT_BIAS]

    _traverse_dicts(feature, upgrade_use_bias)
    return feature


@register_config_transformation("0.5", ["input_features", "output_features"])
def _upgrade_feature(feature: FeatureConfigDict) -> FeatureConfigDict:
    """Upgrades feature config (in-place)"""
    if feature.get(TYPE) == "numerical":
        warnings.warn('Feature type "numerical" renamed to "number" and will be removed in v0.6', DeprecationWarning)
        feature[TYPE] = NUMBER
    if feature.get(TYPE) == AUDIO:
        if PREPROCESSING in feature:
            feature[PREPROCESSING] = upgrade_audio_preprocessing(feature[PREPROCESSING])
        warnings.warn(
            "Parameters specified at the `audio_feature` parameter level have been unnested and should now "
            "be specified at the preprocessing level. Support for `audio_feature` will be removed in v0.7",
            DeprecationWarning,
        )
    return feature


def upgrade_audio_preprocessing(preproc_dict: PreprocessingConfigDict) -> PreprocessingConfigDict:
    if "audio_feature" in preproc_dict:
        for k, v in preproc_dict["audio_feature"].items():
            preproc_dict[k] = v
        del preproc_dict["audio_feature"]
    return preproc_dict


@register_config_transformation("0.6", ["input_features"])
def _upgrade_encoder_params(feature: FeatureConfigDict) -> FeatureConfigDict:
    return _upgrade_encoder_decoder_params(feature, True)


@register_config_transformation("0.6", ["output_features"])
def _upgrade_decoder_params(feature: FeatureConfigDict) -> FeatureConfigDict:
    return _upgrade_encoder_decoder_params(feature, False)


def _upgrade_encoder_decoder_params(feature: FeatureConfigDict, input_feature: bool) -> FeatureConfigDict:
    """
    This function nests un-nested encoder/decoder parameters to conform with the new config structure for 0.6
    Args:
        feature (Dict): Feature to nest encoder/decoder params for.
        input_feature (Bool): Whether this feature is an input feature or not.
    """
    if TYPE not in feature:
        return feature

    try:
        if input_feature:
            module_type = ENCODER
            feature_cls = get_from_registry(feature[TYPE], get_input_type_registry())
        else:
            module_type = DECODER
            feature_cls = get_from_registry(feature[TYPE], get_output_type_registry())
    except ValueError:
        logging.exception("Failed to obtain encoder / decoder from registry")
        return feature

    feature_schema_cls = feature_cls.get_schema_cls()
    feature_keys = feature_schema_cls.get_valid_field_names()

    # These keys have been renamed from the form below to `fc_<key>` in the new config
    fc_layer_keys = [
        "fc_layers",
        "output_size",
        "use_bias",
        "weights_initializer",
        "bias_initializer",
        "norm",
        "norm_params",
        "activation",
        "dropout",
    ]

    module = feature.get(module_type, {})

    warn = False
    if isinstance(module, str):
        module = {TYPE: module}
        feature[module_type] = module
        warn = True

    nested_params = []
    for k, v in feature.items():
        if k not in feature_keys:
            module[k] = v
            if k in fc_layer_keys and module_type == DECODER:
                module[f"fc_{k}"] = v
            nested_params.append(k)
            warn = True

    if module:
        if module_type in feature:
            feature[module_type].update(module)
        else:
            feature[module_type] = module

    for k in nested_params:
        del feature[k]

    if warn:
        warnings.warn(
            f"{module_type} specific parameters should now be nested within a dictionary under the '{module_type}' "
            f"parameter. Support for un-nested {module_type} specific parameters will be removed in v0.7",
            DeprecationWarning,
        )
    return feature


@register_config_transformation("0.5", ["hyperopt"])
def _upgrade_hyperopt(hyperopt: HyperoptConfigDict) -> HyperoptConfigDict:
    """Upgrades hyperopt config (in-place)"""
    # check for use of legacy "training" reference, if any found convert to "trainer"
    if PARAMETERS in hyperopt:
        hparams = hyperopt[PARAMETERS]
        for k, v in list(hparams.items()):
            substr = "training."
            if k.startswith(substr):
                warnings.warn(
                    'Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning
                )
                hparams["trainer." + k[len(substr) :]] = v
                del hparams[k]

    # check for legacy parameters in "executor"
    if EXECUTOR in hyperopt:
        hpexecutor = hyperopt[EXECUTOR]
        executor_type = hpexecutor.get(TYPE, None)
        if executor_type is not None and executor_type != RAY:
            warnings.warn(
                f'executor type "{executor_type}" not supported, converted to "ray" will be flagged as error '
                "in v0.6",
                DeprecationWarning,
            )
            hpexecutor[TYPE] = RAY

        # if search_alg not at top level and is present in executor, promote to top level
        if SEARCH_ALG in hpexecutor:
            # promote only if not in top-level, otherwise use current top-level
            if SEARCH_ALG not in hyperopt:
                hyperopt[SEARCH_ALG] = hpexecutor[SEARCH_ALG]
                if isinstance(hyperopt[SEARCH_ALG], str):
                    hyperopt[SEARCH_ALG] = {TYPE: hyperopt[SEARCH_ALG]}
            del hpexecutor[SEARCH_ALG]
    else:
        warnings.warn(
            'Missing "executor" section, adding "ray" executor will be flagged as error in v0.6', DeprecationWarning
        )
        hyperopt[EXECUTOR] = {TYPE: RAY}

    # check for legacy "sampler" section
    if SAMPLER in hyperopt:
        warnings.warn(
            f'"{SAMPLER}" is no longer supported, converted to "{SEARCH_ALG}". "{SAMPLER}" will be flagged as '
            "error in v0.6",
            DeprecationWarning,
        )
        if SEARCH_ALG in hyperopt[SAMPLER]:
            if SEARCH_ALG not in hyperopt:
                hyperopt[SEARCH_ALG] = hyperopt[SAMPLER][SEARCH_ALG]
                if isinstance(hyperopt[SEARCH_ALG], str):
                    hyperopt[SEARCH_ALG] = {TYPE: hyperopt[SEARCH_ALG]}
                warnings.warn('Moved "search_alg" to hyperopt config top-level', DeprecationWarning)

        # if num_samples or scheduler exist in SAMPLER move to EXECUTOR Section
        if NUM_SAMPLES in hyperopt[SAMPLER] and NUM_SAMPLES not in hyperopt[EXECUTOR]:
            hyperopt[EXECUTOR][NUM_SAMPLES] = hyperopt[SAMPLER][NUM_SAMPLES]
            warnings.warn('Moved "num_samples" from "sampler" to "executor"', DeprecationWarning)

        if SCHEDULER in hyperopt[SAMPLER] and SCHEDULER not in hyperopt[EXECUTOR]:
            hyperopt[EXECUTOR][SCHEDULER] = hyperopt[SAMPLER][SCHEDULER]
            warnings.warn('Moved "scheduler" from "sampler" to "executor"', DeprecationWarning)

        if SCHEDULER in hyperopt[EXECUTOR] and len(hyperopt[EXECUTOR][SCHEDULER].keys()) == 0:
            del hyperopt[EXECUTOR][SCHEDULER]

        # remove legacy section
        del hyperopt[SAMPLER]

    if SEARCH_ALG not in hyperopt:
        # make top-level as search_alg, if missing put in default value
        hyperopt[SEARCH_ALG] = {TYPE: "variant_generator"}
        warnings.warn(
            'Missing "search_alg" at hyperopt top-level, adding in default value, will be flagged as error ' "in v0.6",
            DeprecationWarning,
        )
    return hyperopt


@register_config_transformation("0.5", ["trainer"])
def _upgrade_trainer(trainer: TrainerConfigDict) -> TrainerConfigDict:
    """Upgrades trainer config (in-place)"""
    eval_batch_size = trainer.get(EVAL_BATCH_SIZE)
    if eval_batch_size == 0:
        warnings.warn(
            "`trainer.eval_batch_size` value `0` changed to `None`, will be unsupported in v0.6", DeprecationWarning
        )
        trainer[EVAL_BATCH_SIZE] = None
    return trainer


@register_config_transformation("0.5")
def _upgrade_preprocessing_defaults(config: ModelConfigDict) -> ModelConfigDict:
    """Move feature-specific preprocessing parameters into defaults in config (in-place)"""
    type_specific_preprocessing_params = dict()

    # If preprocessing section specified and it contains feature specific preprocessing parameters,
    # make a copy and delete it from the preprocessing section
    for parameter in list(config.get(PREPROCESSING, {})):
        if parameter in get_base_type_registry():
            warnings.warn(
                f"Moving preprocessing configuration for `{parameter}` feature type from `preprocessing` section"
                " to `defaults` section in Ludwig config. This will be unsupported in v0.8.",
                DeprecationWarning,
            )
            type_specific_preprocessing_params[parameter] = config[PREPROCESSING].pop(parameter)

        if parameter == "numerical":
            warnings.warn(
                f"Moving preprocessing configuration for `{parameter}` feature type from `preprocessing` section"
                " to `defaults` section in Ludwig config. This will be unsupported in v0.8.",
                DeprecationWarning,
            )
            type_specific_preprocessing_params[NUMBER] = config[PREPROCESSING].pop(parameter)

    # Delete empty preprocessing section if no other preprocessing parameters specified
    if PREPROCESSING in config and not config[PREPROCESSING]:
        del config[PREPROCESSING]

    # Update defaults with the default feature specific preprocessing parameters
    defaults = config.get(DEFAULTS, {})
    for feature_type, preprocessing_param in type_specific_preprocessing_params.items():
        if PREPROCESSING in preprocessing_param:
            preprocessing_param = preprocessing_param[PREPROCESSING]

        if feature_type == AUDIO:
            preprocessing_param = upgrade_audio_preprocessing(preprocessing_param)

        # If defaults was empty, then create a new key with feature type
        if feature_type not in defaults:
            defaults[feature_type] = {PREPROCESSING: preprocessing_param}
        # Feature type exists but preprocessing hasn't be specified
        elif PREPROCESSING not in defaults[feature_type]:
            defaults[feature_type][PREPROCESSING] = preprocessing_param
        # Update default feature specific preprocessing with parameters from config
        else:
            defaults[feature_type][PREPROCESSING].update(
                merge_dict(defaults[feature_type][PREPROCESSING], preprocessing_param)
            )

    if defaults:
        config[DEFAULTS] = defaults

    return config


@register_config_transformation("0.5", "preprocessing")
def _upgrade_preprocessing_split(preprocessing: PreprocessingConfigDict) -> PreprocessingConfigDict:
    """Upgrade split related parameters in preprocessing."""
    split_params = {}

    force_split = preprocessing.pop(FORCE_SPLIT, None)
    split_probabilities = preprocessing.pop(SPLIT_PROBABILITIES, None)
    stratify = preprocessing.pop(STRATIFY, None)

    if split_probabilities is not None:
        split_params[PROBABILITIES] = split_probabilities
        warnings.warn(
            "`preprocessing.split_probabilities` has been replaced by `preprocessing.split.probabilities`, "
            "will be flagged as error in v0.7",
            DeprecationWarning,
        )

    if stratify is not None:
        split_params[TYPE] = STRATIFY
        split_params[COLUMN] = stratify
        warnings.warn(
            "`preprocessing.stratify` has been replaced by `preprocessing.split.column` "
            'when setting `preprocessing.split.type` to "stratify", '
            "will be flagged as error in v0.7",
            DeprecationWarning,
        )

    if force_split is not None:
        warnings.warn(
            "`preprocessing.force_split` has been replaced by `preprocessing.split.type`, "
            "will be flagged as error in v0.7",
            DeprecationWarning,
        )

        if TYPE not in split_params:
            split_params[TYPE] = RANDOM

    if split_params:
        preprocessing[SPLIT] = split_params

    if AUDIO in preprocessing:
        if "audio_feature" in preprocessing[AUDIO]:
            for k, v in preprocessing[AUDIO]["audio_feature"].items():
                preprocessing[AUDIO][k] = v
            del preprocessing[AUDIO]["audio_feature"]
        warnings.warn(
            "Parameters specified at the `audio_feature` parameter level have been unnested and should now "
            "be specified at the preprocessing level. Support for `audio_feature` will be removed in v0.7",
            DeprecationWarning,
        )
    return preprocessing


@register_config_transformation("0.5")
def update_training(config: ModelConfigDict) -> ModelConfigDict:
    if TRAINING in config:
        warnings.warn('Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning)
        config[TRAINER] = config[TRAINING]
        del config[TRAINING]
    return config


@register_config_transformation("0.6")
def upgrade_missing_value_strategy(config: ModelConfigDict) -> ModelConfigDict:
    for input_feature in config.get(INPUT_FEATURES, []):
        if _is_old_missing_value_strategy(input_feature):
            _update_old_missing_value_strategy(input_feature)

    for output_feature in config.get(OUTPUT_FEATURES, []):
        if _is_old_missing_value_strategy(output_feature):
            _update_old_missing_value_strategy(output_feature)

    for feature, feature_defaults in config.get(DEFAULTS, {}).items():
        if _is_old_missing_value_strategy(feature_defaults):
            _update_old_missing_value_strategy(config.get(DEFAULTS).get(feature))

    return config


@register_config_transformation("0.6", ["trainer"])
def _upgrade_max_batch_size(trainer: TrainerConfigDict) -> TrainerConfigDict:
    if "increase_batch_size_on_plateau_max" in trainer:
        warnings.warn(
            'Config param "increase_batch_size_on_plateau_max" renamed to "max_batch_size" and will be '
            "removed in v0.8",
            DeprecationWarning,
        )
        increase_batch_size_on_plateau_max_val = trainer.pop("increase_batch_size_on_plateau_max")
        if "max_batch_size" in trainer:
            warnings.warn('"max_batch_size" config param already set. Discarding "increase_batch_size_on_plateau_max".')
        else:
            warnings.warn(
                f'Setting "max_batch_size" config param to "increase_batch_size_on_plateau_max" value '
                f'({increase_batch_size_on_plateau_max_val}) and discarding "increase_batch_size_on_plateau_max"'
            )
            trainer["max_batch_size"] = increase_batch_size_on_plateau_max_val
    return trainer


@register_config_transformation("0.6")
def remove_trainer_type(config: ModelConfigDict) -> ModelConfigDict:
    # LLM Model types support different trainer types
    if config.get("model_type", None) == "llm":
        return config

    if TYPE in config.get("trainer", {}):
        warnings.warn(
            "Config param `type` has been removed from the trainer. The trainer type is determined by the top level "
            " `model_type` parameter. Support for the `type` params in trainer will be removed in v0.8",
            DeprecationWarning,
        )
        del config["trainer"][TYPE]

    return config


@register_config_transformation("0.7", ["trainer"])
def learning_rate_scheduler(trainer: TrainerConfigDict) -> TrainerConfigDict:
    key_mapping = {
        "reduce_learning_rate_on_plateau": "reduce_on_plateau",
        "reduce_learning_rate_on_plateau_patience": "reduce_on_plateau_patience",
        "reduce_learning_rate_on_plateau_rate": "reduce_on_plateau_rate",
        "reduce_learning_rate_eval_metric": "reduce_eval_metric",
        "reduce_learning_rate_eval_split": "reduce_eval_split",
        "decay": "decay",
        "decay_steps": "decay_steps",
        "decay_rate": "decay_rate",
        "staircase": "staircase",
        "learning_rate_warmup_epochs": "warmup_evaluations",
    }

    lr_scheduler = trainer.get("learning_rate_scheduler", {})
    for old_key, new_key in key_mapping.items():
        if old_key in trainer:
            warnings.warn(
                f"Config param `trainer.{old_key}` has been moved to `trainer.learning_rate_scheduler.{new_key}`.",
                DeprecationWarning,
            )
            if new_key in lr_scheduler:
                warnings.warn(
                    f"`trainer.learning_rate_scheduler.{new_key}` config param already set. "
                    f"Discarding `trainer.{old_key}`."
                )
            else:
                value = trainer[old_key]
                if old_key == "decay" and isinstance(value, bool):
                    # Decay has changed from a bool to an optional enum
                    lr_scheduler[new_key] = "exponential" if value else None
                elif old_key == "reduce_learning_rate_on_plateau":
                    lr_scheduler[new_key] = int(value)
                else:
                    lr_scheduler[new_key] = value
            del trainer[old_key]

    if lr_scheduler:
        trainer["learning_rate_scheduler"] = lr_scheduler

    return trainer


@register_config_transformation("0.7", ["input_features"])
def _upgrade_legacy_image_encoders(feature: FeatureConfigDict) -> FeatureConfigDict:
    if feature.get(TYPE) != IMAGE:
        return feature

    encoder_mapping = {
        "resnet": "_resnet_legacy",
        "vit": "_vit_legacy",
    }

    encoder = feature.get(ENCODER, {})
    encoder_type = encoder.get(TYPE)
    if encoder_type not in encoder_mapping:
        return feature

    # For this version of Ludwig, only ECD supported these encoders.
    new_encoder_cls = get_encoder_cls(MODEL_ECD, feature[TYPE], encoder_type)
    new_encoder_fields = new_encoder_cls.get_valid_field_names()

    legacy_encoder_cls = get_encoder_cls(MODEL_ECD, feature[TYPE], encoder_mapping[encoder_type])
    legacy_encoder_fields = legacy_encoder_cls.get_valid_field_names()

    user_fields = set(encoder.keys())
    user_fields.remove(TYPE)

    removed_fields = legacy_encoder_fields.difference(new_encoder_fields)
    added_fields = new_encoder_fields.difference(legacy_encoder_fields)

    user_legacy_fields = user_fields.intersection(removed_fields)
    user_new_fields = user_fields.intersection(added_fields)
    if len(user_legacy_fields) > 0:
        if len(user_new_fields) > 0:
            raise ValueError(
                f"Intended encoder type is ambiguous. "
                f"Provided encoder fields matching encoder '{encoder_type}' {user_new_fields} and "
                f"legacy encoder '{encoder_mapping[encoder_type]}' {user_legacy_fields}. "
                f"Please remove features unique to one of these encoder types from your configuration."
            )

        warnings.warn(
            f"Encoder '{encoder_type}' with params '{user_legacy_fields}' has been renamed to "
            f"'{encoder_mapping[encoder_type]}'. Please upgrade your config to use the new '{encoder_type}' as "
            f"support for '{encoder_mapping[encoder_type]}' is not guaranteed past v0.8.",
            DeprecationWarning,
        )

        # User provided legacy fields and no new fields, so we assume they intended to use the legacy encoder
        encoder[TYPE] = encoder_mapping[encoder_type]

    return feature


@register_config_transformation("0.7")
def upgrade_missing_hyperopt(config: ModelConfigDict) -> ModelConfigDict:
    hyperopt = config.get(HYPEROPT)
    if hyperopt == {}:
        # This is a deprecated form of providing a missing hyperopt section, as it violates the schema definition
        warnings.warn(
            "Config section `hyperopt: {}` is deprecated, please set `hyperopt: null` to disable hyperopt.",
            DeprecationWarning,
        )
        del config[HYPEROPT]
    return config


@register_config_transformation("0.7")
def upgrade_defaults_config_for_gbm(config: ModelConfigDict) -> ModelConfigDict:
    model_type = config.get(MODEL_TYPE, "")
    if model_type != MODEL_GBM:
        return config

    defaults_ref = config.get(DEFAULTS, {})
    defaults = copy.deepcopy(defaults_ref)
    gbm_feature_types = GBMDefaultsConfig.Schema().fields.keys()
    for feature_type in defaults_ref:
        # GBM only supports binary, number, category and text features
        if feature_type not in gbm_feature_types:
            del defaults[feature_type]
            continue

        # Remove decoder and loss from defaults since they only apply to ECD
        defaults[feature_type].pop("decoder", None)
        defaults[feature_type].pop("loss", None)
    config[DEFAULTS] = defaults
    return config


@register_config_transformation("0.7", "defaults")
def remove_extra_type_param_in_defaults_config(defaults: FeatureTypeDefaultsDict) -> FeatureTypeDefaultsDict:
    """Fixes a bug introduced before 0.7.3.

    [1] and subsequent refactors accidentally introduced a bug where a `type` param was added to every feature in the
    defaults config. It was removed by [2], but made it into one of the patch releases. This transformation removes that
    `type` param from each section of the defaults config if it exists.

    [1]: https://github.com/ludwig-ai/ludwig/pull/3223
    [2]: https://github.com/ludwig-ai/ludwig/pull/3258
    """
    defaults_copy = copy.deepcopy(defaults)
    for feature_type, feature_config in defaults.items():
        if TYPE in feature_config:
            del defaults_copy[feature_type][TYPE]
    return defaults_copy


def upgrade_metadata(metadata: TrainingSetMetadataDict) -> TrainingSetMetadataDict:
    # TODO(travis): stopgap solution, we should make it so we don't need to do this
    # by decoupling config and metadata
    metadata = copy.deepcopy(metadata)
    _upgrade_metadata_missing_values(metadata)
    return metadata


def _upgrade_metadata_missing_values(metadata: TrainingSetMetadataDict):
    for k, v in metadata.items():
        if isinstance(v, dict) and _is_old_missing_value_strategy(v):
            _update_old_missing_value_strategy(v)
        elif isinstance(v, dict) and _is_image_feature(v):
            _update_old_image_preprocessing(v)


def _update_old_missing_value_strategy(feature_config: FeatureConfigDict):
    missing_value_strategy = feature_config.get(PREPROCESSING).get(MISSING_VALUE_STRATEGY)
    replacement_strategy = "bfill" if missing_value_strategy == "backfill" else "ffill"
    feature_name = feature_config.get(NAME)
    warnings.warn(
        f"Using `{replacement_strategy}` instead of `{missing_value_strategy}` as the missing value strategy"
        f" for `{feature_name}`. These are identical. `{missing_value_strategy}` will be removed in v0.8",
        DeprecationWarning,
    )
    feature_config[PREPROCESSING].update({MISSING_VALUE_STRATEGY: replacement_strategy})


def _is_old_missing_value_strategy(feature_config: FeatureConfigDict):
    if PREPROCESSING not in feature_config:
        return False
    missing_value_strategy = feature_config.get(PREPROCESSING).get(MISSING_VALUE_STRATEGY, None)
    if not missing_value_strategy or missing_value_strategy not in ("backfill", "pad"):
        return False
    return True


def _is_image_feature(feature_config: FeatureConfigDict):
    preproc = feature_config.get(PREPROCESSING, {})
    return HEIGHT in preproc and WIDTH in preproc


def _update_old_image_preprocessing(feature_config: FeatureConfigDict):
    preprocessing = feature_config.get(PREPROCESSING)
    if not preprocessing:
        return
    preprocessing["standardize_image"] = preprocessing.get("standardize_image")
