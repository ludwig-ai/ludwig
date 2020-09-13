import logging
import os
import os.path
import tempfile
from collections import namedtuple

import pytest
import yaml

from ludwig.experiment import kfold_cross_validate, full_kfold_cross_validate
from ludwig.utils.data_utils import load_json
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

FeaturesToUse = namedtuple('FeaturesToUse', 'input_features output_features')

FEATURES_TO_TEST = [
    FeaturesToUse(
        # input feature
        [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ],
        # output feature
        [
            numerical_feature()
        ]
    ),
    FeaturesToUse(
        # input feature
        [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ],
        # output feature
        [
            binary_feature()
        ]
    ),
    FeaturesToUse(
        # input feature
        [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ],
        # output feature
        [
            category_feature(vocab_size=2, reduce_input='sum')
        ]
    ),
    FeaturesToUse(
        # input feature
        [
            sequence_feature(
                min_len=5,
                max_len=10,
                encoder='rnn',
                cell_type='lstm',
                reduce_output=None
            )
        ],
        # output feature
        [
            sequence_feature(
                min_len=5,
                max_len=10,
                decoder='generator',
                cell_type='lstm',
                attention='bahdanau',
                reduce_input=None
            )
        ]
    ),
    FeaturesToUse(
        # input feature
        [
            sequence_feature(
                min_len=5,
                max_len=10,
                encoder='rnn',
                cell_type='lstm',
                reduce_output=None
            )

        ],
        # output feature
        [
            sequence_feature(
                max_len=10,
                decoder='tagger',
                reduce_input=None
            )
        ]
    )
]


@pytest.mark.parametrize('features_to_use', FEATURES_TO_TEST)
def test_kfold_cv_cli(features_to_use):
    # k-fold cross validation cli
    num_folds = 3

    # setup temporary directory to run test
    with tempfile.TemporaryDirectory() as tmpdir:

        training_data_fp = os.path.join(tmpdir, 'train.csv')
        model_definition_fp = os.path.join(tmpdir, 'model_definition.yaml')
        results_dir = os.path.join(tmpdir, 'results')
        statistics_fp = os.path.join(results_dir,
                                     'kfold_training_statistics.json')
        indices_fp = os.path.join(results_dir, 'kfold_split_indices.json')

        # generate synthetic data for the test
        input_features = features_to_use.input_features

        output_features = features_to_use.output_features

        generate_data(input_features, output_features, training_data_fp)

        # generate model definition file
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        with open(model_definition_fp, 'w') as f:
            yaml.dump(model_definition, f)

        # run k-fold cv
        full_kfold_cross_validate(
            k_fold=num_folds,
            model_definition_file=model_definition_fp,
            data_csv=training_data_fp,
            output_directory=results_dir,
            logging_level='warn'
        )

        # check for expected results
        # check for existence and structure of statistics file
        assert os.path.isfile(statistics_fp)

        # check for required keys
        cv_statistics = load_json(statistics_fp)
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in cv_statistics

        # check for existence and structure of split indices file
        assert os.path.isfile(indices_fp)

        # check for required keys
        cv_indices = load_json(indices_fp)
        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in cv_indices


def test_kfold_cv_api_from_file():
    # k-fold_cross_validate api with model_definition_file
    num_folds = 3

    # setup temporary directory to run test
    with tempfile.TemporaryDirectory() as tmpdir:

        # setup required data structures for test
        training_data_fp = os.path.join(tmpdir, 'train.csv')
        model_definition_fp = os.path.join(tmpdir, 'model_definition.yaml')

        # generate synthetic data for the test
        input_features = [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ]

        output_features = [
            category_feature(vocab_size=2, reduce_input='sum')
        ]

        generate_data(input_features, output_features, training_data_fp)

        # generate model definition file
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        with open(model_definition_fp, 'w') as f:
            yaml.dump(model_definition, f)

        # test kfold_cross_validate api with model definition file

        # execute k-fold cross validation run
        (
            kfold_cv_stats,
            kfold_split_indices
         ) = kfold_cross_validate(
            3,
            model_definition_file=model_definition_fp,
            data_csv=training_data_fp
        )

        # correct structure for results from kfold cv
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in kfold_cv_stats

        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in kfold_split_indices

def test_kfold_cv_api_in_memory():
    # k-fold_cross_validate api with in-memory model defintion
    num_folds = 3

    # setup temporary directory to run test
    with tempfile.TemporaryDirectory() as tmpdir:

        # setup required data structures for test
        training_data_fp = os.path.join(tmpdir, 'train.csv')

        # generate synthetic data for the test
        input_features = [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ]

        output_features = [
            numerical_feature()
        ]

        generate_data(input_features, output_features, training_data_fp)

        # generate model definition file
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        # test kfold_cross_validate api with model definition in-memory

        # execute k-fold cross validation run
        (
            kfold_cv_stats,
            kfold_split_indices
        ) = kfold_cross_validate(
             3,
            model_definition=model_definition,
            data_csv=training_data_fp
        )

        # correct structure for results from kfold cv
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in kfold_cv_stats

        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in kfold_split_indices
