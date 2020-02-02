import logging
import os
import os.path
import tempfile

import yaml

from ludwig.experiment import experiment_kfold_cross_validate
from ludwig.utils.data_utils import load_json
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


def test_kfold_cv():
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

        # run k-fold cv
        experiment_kfold_cross_validate(
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
