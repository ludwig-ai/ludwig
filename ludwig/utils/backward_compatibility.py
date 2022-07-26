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

import warnings
from typing import Any, Callable, Dict

from ludwig.constants import (
    AUDIO,
    BIAS,
    COLUMN,
    CONV_BIAS,
    CONV_USE_BIAS,
    DECODER,
    DEFAULT_BIAS,
    DEFAULT_USE_BIAS,
    ENCODER,
    EVAL_BATCH_SIZE,
    EXECUTOR,
    FORCE_SPLIT,
    HYPEROPT,
    INPUT_FEATURE_KEYS,
    INPUT_FEATURES,
    NUM_SAMPLES,
    NUMBER,
    OUTPUT_FEATURE_KEYS,
    OUTPUT_FEATURES,
    PARAMETERS,
    PREPROCESSING,
    PROBABILITIES,
    RANDOM,
    RAY,
    SAMPLER,
    SCHEDULER,
    SEARCH_ALG,
    SPLIT,
    SPLIT_PROBABILITIES,
    STRATIFY,
    TRAINER,
    TRAINING,
    TYPE,
    USE_BIAS,
)


def _traverse_dicts(config: Any, f: Callable[[Dict], None]):
    """Applies function f to every dictionary contained in config.

    f should in-place modify the config dict. f will be called on leaves first, root last.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            _traverse_dicts(v, f)
        f(config)
    elif isinstance(config, list):
        for v in config:
            _traverse_dicts(v, f)


def _upgrade_use_bias(config):
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


def _upgrade_feature(feature: Dict[str, Any]):
    """Upgrades feature config (in-place)"""
    if feature.get(TYPE) == "numerical":
        warnings.warn('Feature type "numerical" renamed to "number" and will be removed in v0.6', DeprecationWarning)
        feature[TYPE] = NUMBER
    if feature.get(TYPE) == AUDIO:
        if PREPROCESSING in feature:
            if "audio_feature" in feature[PREPROCESSING]:
                for k, v in feature[PREPROCESSING]["audio_feature"].items():
                    feature[PREPROCESSING][k] = v
                del feature[PREPROCESSING]["audio_feature"]
        warnings.warn(
            "Parameters specified at the `audio_feature` parameter level have been unnested and should now "
            "be specified at the preprocessing level. Support for `audio_feature` will be removed in v0.7",
            DeprecationWarning,
        )
    _traverse_dicts(feature, _upgrade_use_bias)


def _upgrade_encoder_decoder_params(feature: Dict[str, Any], input_feature: bool):
    """
    This function nests un-nested encoder/decoder parameters to conform with the new config structure for 0.6
    Args:
        feature (Dict): Feature to nest encoder/decoder params for.
        input_feature (Bool): Whether this feature is an input feature or not.

    Returns:

    """
    if input_feature:
        encoder = feature.get(ENCODER, {})
        if isinstance(encoder, str):
            encoder = {TYPE: encoder}
            feature[ENCODER] = encoder

        nested_params = []
        for k, v in feature.items():
            if k not in INPUT_FEATURE_KEYS:
                encoder[k] = v
                nested_params.append(k)

        if ENCODER in feature:
            feature[ENCODER].update(encoder)
        else:
            feature[ENCODER] = encoder

        for k in nested_params:
            del feature[k]
        warnings.warn(
            "Encoder specific parameters should now be nested within a dictionary under the 'encoder' parameter. "
            "Support for un-nested encoder specific parameters will be removed in v0.7",
            DeprecationWarning,
        )
    else:
        decoder = feature.get(DECODER, {})
        if isinstance(decoder, str):
            decoder = {TYPE: decoder}
            feature[DECODER] = decoder

        nested_params = []
        for k, v in feature.items():
            if k not in OUTPUT_FEATURE_KEYS:
                decoder[k] = v
                nested_params.append(k)

        if DECODER in feature:
            feature[DECODER].update(decoder)
        else:
            feature[DECODER] = decoder

        for k in nested_params:
            if k in nested_params:
                del feature[k]
        warnings.warn(
            "Decoder specific parameters should now be nested within a dictionary under the 'decoder' parameter. "
            "Support for un-nested decoder specific parameters will be removed in v0.7",
            DeprecationWarning,
        )


def _upgrade_hyperopt(hyperopt: Dict[str, Any]):
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
                warnings.warn('Moved "search_alg" to hyperopt config top-level', DeprecationWarning)

        # if num_samples or scheduler exist in SAMPLER move to EXECUTOR Section
        if NUM_SAMPLES in hyperopt[SAMPLER] and NUM_SAMPLES not in hyperopt[EXECUTOR]:
            hyperopt[EXECUTOR][NUM_SAMPLES] = hyperopt[SAMPLER][NUM_SAMPLES]
            warnings.warn('Moved "num_samples" from "sampler" to "executor"', DeprecationWarning)

        if SCHEDULER in hyperopt[SAMPLER] and SCHEDULER not in hyperopt[EXECUTOR]:
            hyperopt[EXECUTOR][SCHEDULER] = hyperopt[SAMPLER][SCHEDULER]
            warnings.warn('Moved "scheduler" from "sampler" to "executor"', DeprecationWarning)

        # remove legacy section
        del hyperopt[SAMPLER]

    if SEARCH_ALG not in hyperopt:
        # make top-level as search_alg, if missing put in default value
        hyperopt[SEARCH_ALG] = {TYPE: "variant_generator"}
        warnings.warn(
            'Missing "search_alg" at hyperopt top-level, adding in default value, will be flagged as error ' "in v0.6",
            DeprecationWarning,
        )


def _upgrade_trainer(trainer: Dict[str, Any]):
    """Upgrades trainer config (in-place)"""
    eval_batch_size = trainer.get(EVAL_BATCH_SIZE)
    if eval_batch_size == 0:
        warnings.warn(
            "`trainer.eval_batch_size` value `0` changed to `None`, will be unsupported in v0.6", DeprecationWarning
        )
        trainer[EVAL_BATCH_SIZE] = None


def _upgrade_preprocessing(preprocessing: Dict[str, Any]):
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


def upgrade_deprecated_fields(config: Dict[str, Any]):
    """Updates config (in-place) to use fields from earlier versions of Ludwig.

    Logs deprecation warnings
    """
    if TRAINING in config:
        warnings.warn('Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning)
        config[TRAINER] = config[TRAINING]
        del config[TRAINING]

    for feature in config.get(INPUT_FEATURES, []) + config.get(OUTPUT_FEATURES, []):
        _upgrade_feature(feature)

    for feature in config.get(INPUT_FEATURES, []):
        _upgrade_encoder_decoder_params(feature, True)

    for feature in config.get(OUTPUT_FEATURES, []):
        _upgrade_encoder_decoder_params(feature, False)

    if HYPEROPT in config:
        _upgrade_hyperopt(config[HYPEROPT])

    if TRAINER in config:
        _upgrade_trainer(config[TRAINER])

    if PREPROCESSING in config:
        _upgrade_preprocessing(config[PREPROCESSING])
