#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
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
import logging

from ludwig.constants import *
from ludwig.features.feature_registries import (base_type_registry,
                                                input_type_registry,
                                                output_type_registry)
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.utils.misc_utils import (get_from_registry, merge_dict,
                                     set_default_value)

logger = logging.getLogger(__name__)

default_random_seed = 42

default_preprocessing_force_split = False
default_preprocessing_split_probabilities = (0.7, 0.1, 0.2)
default_preprocessing_stratify = None

default_preprocessing_parameters = {
    'force_split': default_preprocessing_force_split,
    'split_probabilities': default_preprocessing_split_probabilities,
    'stratify': default_preprocessing_stratify
}
default_preprocessing_parameters.update({
    name: base_type.preprocessing_defaults for name, base_type in
    base_type_registry.items()
})

default_combiner_type = 'concat'

default_training_params = {
    'optimizer': {TYPE: 'adam'},
    'epochs': 100,
    'regularizer': 'l2',
    'regularization_lambda': 0,
    'learning_rate': 0.001,
    'batch_size': 128,
    'eval_batch_size': 0,
    'early_stop': 5,
    'reduce_learning_rate_on_plateau': 0,
    'reduce_learning_rate_on_plateau_patience': 5,
    'reduce_learning_rate_on_plateau_rate': 0.5,
    'increase_batch_size_on_plateau': 0,
    'increase_batch_size_on_plateau_patience': 5,
    'increase_batch_size_on_plateau_rate': 2,
    'increase_batch_size_on_plateau_max': 512,
    'decay': False,
    'decay_steps': 10000,
    'decay_rate': 0.96,
    'staircase': False,
    'gradient_clipping': None,
    'validation_field': COMBINED,
    'validation_metric': LOSS,
    'bucketing_field': None,
    'learning_rate_warmup_epochs': 1
}

default_optimizer_params_registry = {
    'sgd': {},
    'stochastic_gradient_descent': {},
    'gd': {},
    'gradient_descent': {},
    'adam': {
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08
    },
    'adadelta': {
        'rho': 0.95,
        'epsilon': 1e-08
    },
    'adagrad': {
        'initial_accumulator_value': 0.1
    },
    'adamax': {},
    'ftrl': {
        'learning_rate_power': -0.5,
        'initial_accumulator_value': 0.1,
        'l1_regularization_strength': 0.0,
        'l2_regularization_strength': 0.0
    },
    'nadam': {},
    'rmsprop': {
        'decay': 0.9,
        'momentum': 0.0,
        'epsilon': 1e-10,
        'centered': False
    }
}
default_optimizer_params_registry['stochastic_gradient_descent'] = (
    default_optimizer_params_registry['sgd']
)
default_optimizer_params_registry['gd'] = (
    default_optimizer_params_registry['sgd']
)
default_optimizer_params_registry['gradient_descent'] = (
    default_optimizer_params_registry['sgd']
)


def get_default_optimizer_params(optimizer_type):
    if optimizer_type in default_optimizer_params_registry:
        return default_optimizer_params_registry[optimizer_type]
    else:
        raise ValueError('Incorrect optimizer type: ' + optimizer_type)


def _perform_sanity_checks(config):
    assert 'input_features' in config, (
        'config does not define any input features'
    )

    assert 'output_features' in config, (
        'config does not define any output features'
    )

    assert isinstance(config['input_features'], list), (
        'Ludwig expects input features in a list. Check your model '
        'config format'
    )

    assert isinstance(config['output_features'], list), (
        'Ludwig expects output features in a list. Check your model '
        'config format'
    )

    assert len(config['input_features']) > 0, (
        'config needs to have at least one input feature'
    )

    assert len(config['output_features']) > 0, (
        'config needs to have at least one output feature'
    )

    if TRAINING in config:
        assert isinstance(config[TRAINING], dict), (
            'There is an issue while reading the training section of the '
            'config. The parameters are expected to be'
            'read as a dictionary. Please check your config format.'
        )

    if 'preprocessing' in config:
        assert isinstance(config['preprocessing'], dict), (
            'There is an issue while reading the preprocessing section of the '
            'config. The parameters are expected to be read'
            'as a dictionary. Please check your config format.'
        )

    if 'combiner' in config:
        assert isinstance(config['combiner'], dict), (
            'There is an issue while reading the combiner section of the '
            'config. The parameters are expected to be read'
            'as a dictionary. Please check your config format.'
        )


def _set_feature_column(config: dict) -> None:
    for feature in config['input_features'] + config['output_features']:
        if COLUMN not in feature:
            feature[COLUMN] = feature[NAME]


def _set_proc_column(config: dict) -> None:
    for feature in config['input_features'] + config['output_features']:
        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)


def _merge_hyperopt_with_training(config: dict) -> None:
    if 'hyperopt' not in config or TRAINING not in config:
        return

    scheduler = config['hyperopt'].get('sampler', {}).get('scheduler')
    if not scheduler:
        return

    # Disable early stopping when using a scheduler. We achieve this by setting the parameter
    # to -1, which ensures the condition to apply early stopping is never met.
    training = config[TRAINING]
    early_stop = training.get('early_stop')
    if early_stop is not None and early_stop != -1:
        raise ValueError(
            'Cannot set training parameter `early_stop` when using a hyperopt scheduler. '
            'Unset this parameter in your config.'
        )
    training['early_stop'] = -1

    # At most one of max_t and epochs may be specified by the user, and we set them to be equal to
    # ensure that Ludwig does not stop training before the scheduler has finished the trial.
    max_t = scheduler.get('max_t')
    epochs = training.get('epochs')
    if max_t is not None and epochs is not None and max_t != epochs:
        raise ValueError(
            'Cannot set training parameter `epochs` when using a hyperopt scheduler with `max_t`. '
            'Unset one of these parameters in your config.'
        )
    elif max_t is not None:
        training['epochs'] = max_t
    elif epochs is not None:
        scheduler['max_t'] = epochs


def merge_with_defaults(config):
    _perform_sanity_checks(config)
    _set_feature_column(config)
    _set_proc_column(config)
    _merge_hyperopt_with_training(config)

    # ===== Preprocessing =====
    config['preprocessing'] = merge_dict(
        default_preprocessing_parameters,
        config.get('preprocessing', {})
    )

    stratify = config['preprocessing']['stratify']
    if stratify is not None:
        features = (
                config['input_features'] +
                config['output_features']
        )
        feature_names = set(f[COLUMN] for f in features)
        if stratify not in feature_names:
            logger.warning(
                'Stratify is not among the features. '
                'Cannot establish if it is a binary or category'
            )
        elif ([f for f in features if f[COLUMN] == stratify][0][TYPE]
              not in {BINARY, CATEGORY}):
            raise ValueError('Stratify feature must be binary or category')

    # ===== Training =====
    set_default_value(config, TRAINING, default_training_params)

    for param, value in default_training_params.items():
        set_default_value(config[TRAINING], param,
                          value)

    set_default_value(
        config[TRAINING],
        'validation_metric',
        output_type_registry[config['output_features'][0][
            TYPE]].default_validation_metric
    )

    # ===== Training Optimizer =====
    optimizer = config[TRAINING]['optimizer']
    default_optimizer_params = get_default_optimizer_params(optimizer[TYPE])
    for param in default_optimizer_params:
        set_default_value(optimizer, param, default_optimizer_params[param])

    # ===== Input Features =====
    for input_feature in config['input_features']:
        get_from_registry(input_feature[TYPE],
                          input_type_registry).populate_defaults(input_feature)

    # ===== Combiner =====
    set_default_value(config, 'combiner',
                      {TYPE: default_combiner_type})

    # ===== Output features =====
    for output_feature in config['output_features']:
        get_from_registry(output_feature[TYPE],
                          output_type_registry).populate_defaults(
            output_feature)

    return config
