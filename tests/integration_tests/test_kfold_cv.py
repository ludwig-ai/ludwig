import logging
import os
import os.path
import tempfile
from collections import namedtuple

import pytest
import yaml
from ludwig.api import kfold_cross_validate
from ludwig.experiment import kfold_cross_validate_cli
from ludwig.utils.data_utils import load_json
from tests.integration_tests.utils import binary_feature, create_data_set_to_use
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import text_feature

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
    ),
    FeaturesToUse(
        # input feature
        [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ],
        # output feature
        [
            text_feature()
        ]
    ),
]


@pytest.mark.parametrize('features_to_use', FEATURES_TO_TEST)
def test_kfold_cv_cli(features_to_use: FeaturesToUse):
    # k-fold cross validation cli
    num_folds = 3

    # setup temporary directory to run test
    with tempfile.TemporaryDirectory() as tmpdir:

        training_data_fp = os.path.join(tmpdir, 'train.csv')
        config_fp = os.path.join(tmpdir, 'config.yaml')
        results_dir = os.path.join(tmpdir, 'results')
        statistics_fp = os.path.join(results_dir,
                                     'kfold_training_statistics.json')
        indices_fp = os.path.join(results_dir, 'kfold_split_indices.json')

        # generate synthetic data for the test
        input_features = features_to_use.input_features

        output_features = features_to_use.output_features

        generate_data(input_features, output_features, training_data_fp)

        # generate config file
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        with open(config_fp, 'w') as f:
            yaml.dump(config, f)

        # run k-fold cv
        kfold_cross_validate_cli(
            k_fold=num_folds,
            config_file=config_fp,
            dataset=training_data_fp,
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
    # k-fold_cross_validate api with config_file
    num_folds = 3

    # setup temporary directory to run test
    with tempfile.TemporaryDirectory() as tmpdir:

        # setup required data structures for test
        training_data_fp = os.path.join(tmpdir, 'train.csv')
        config_fp = os.path.join(tmpdir, 'config.yaml')

        # generate synthetic data for the test
        input_features = [
            numerical_feature(normalization='zscore'),
            numerical_feature(normalization='zscore')
        ]

        output_features = [
            category_feature(vocab_size=2, reduce_input='sum')
        ]

        generate_data(input_features, output_features, training_data_fp)

        # generate config file
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        with open(config_fp, 'w') as f:
            yaml.dump(config, f)

        # test kfold_cross_validate api with config file

        # execute k-fold cross validation run
        (
            kfold_cv_stats,
            kfold_split_indices
        ) = kfold_cross_validate(
            3,
            config=config_fp,
            dataset=training_data_fp
        )

        # correct structure for results from kfold cv
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in kfold_cv_stats

        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in kfold_split_indices


def test_kfold_cv_api_in_memory():
    # k-fold_cross_validate api with in-memory config
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

        # generate config file
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        # test kfold_cross_validate api with config in-memory

        # execute k-fold cross validation run
        (
            kfold_cv_stats,
            kfold_split_indices
        ) = kfold_cross_validate(
            3,
            config=config,
            dataset=training_data_fp
        )

        # correct structure for results from kfold cv
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in kfold_cv_stats

        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in kfold_split_indices



DATA_FORMATS_FOR_KFOLDS = [
    'csv', 'df', 'dict', 'excel', 'feather', 'fwf', 'html',
    'json', 'jsonl', 'parquet', 'pickle', 'stata', 'tsv'
]
@pytest.mark.parametrize('data_format', DATA_FORMATS_FOR_KFOLDS)
def test_kfold_cv_dataset_formats(data_format):
    # k-fold_cross_validate api with in-memory config
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
        dataset_to_use = create_data_set_to_use(data_format, training_data_fp)

        # generate config file
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        # test kfold_cross_validate api with config in-memory

        # execute k-fold cross validation run
        (
            kfold_cv_stats,
            kfold_split_indices
        ) = kfold_cross_validate(
            3,
            config=config,
            dataset=dataset_to_use
        )

        # correct structure for results from kfold cv
        for key in ['fold_' + str(i + 1)
                    for i in range(num_folds)] + ['overall']:
            assert key in kfold_cv_stats

        for key in ['fold_' + str(i + 1) for i in range(num_folds)]:
            assert key in kfold_split_indices

