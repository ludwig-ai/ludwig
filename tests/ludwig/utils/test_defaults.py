import copy

import pytest

from ludwig.constants import TRAINING, HYPEROPT
from ludwig.utils.defaults import merge_with_defaults, default_training_params
from tests.integration_tests.utils import binary_feature, category_feature, \
    numerical_feature, text_feature, sequence_feature, vector_feature


HYPEROPT_CONFIG = {
    "parameters": {
        "training.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {
            "space": "randint",
            "lower": 2,
            "upper": 6
        },
        "utterance.cell_type": {
            "space": "grid_search",
            "values": ["rnn", "gru"]
        },
        "utterance.bidirectional": {
            "space": "choice",
            "categories": [True, False]
        },
        "utterance.fc_layers": {
            "space": "choice",
            "categories": [
                [{"fc_size": 512}, {"fc_size": 256}],
                [{"fc_size": 512}],
                [{"fc_size": 256}],
            ]
        }
    },
    "sampler": {"type": "ray"},
    "executor": {"type": "ray"},
    "goal": "minimize"
}

SCHEDULER = {'type': 'async_hyperband', 'time_attr': 'time_total_s'}

default_early_stop = default_training_params['early_stop']


@pytest.mark.parametrize("use_train,use_hyperopt_scheduler", [
    (True,True),
    (False,True),
    (True,False),
    (False,False),
])
def test_merge_with_defaults_early_stop(use_train, use_hyperopt_scheduler):
    all_input_features = [
        binary_feature(),
        category_feature(),
        numerical_feature(),
        text_feature(),
    ]
    all_output_features = [
        category_feature(),
        sequence_feature(),
        vector_feature(),
    ]

    # validate config with all features
    config = {
        'input_features': all_input_features,
        'output_features': all_output_features,
        HYPEROPT: HYPEROPT_CONFIG,
    }
    config = copy.deepcopy(config)

    if use_train:
        config[TRAINING] = {'batch_size': '42'}

    if use_hyperopt_scheduler:
        # hyperopt scheduler cannot be used with early stopping
        config[HYPEROPT]['sampler']['scheduler'] = SCHEDULER

    merged_config = merge_with_defaults(config)

    expected = -1 if use_hyperopt_scheduler else default_early_stop
    assert merged_config[TRAINING]['early_stop'] == expected
