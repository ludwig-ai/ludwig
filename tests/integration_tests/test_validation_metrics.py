import logging
from collections import namedtuple

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import *
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

TestCase = namedtuple('TestCase', 'output_features validation_metrics')


# output_features: output features to test
# validation_metrics: relevant metrics for the output feature


# test for single output feature
@pytest.mark.parametrize(
    'test_case',
    [
        TestCase(
            [numerical_feature()],
            ['loss', 'mean_squared_error', 'mean_absolute_error', 'r2']
        ),
        TestCase(
            [binary_feature()],
            ['loss', 'accuracy']
        ),
        TestCase(
            [category_feature()],
            ['loss', 'accuracy', 'hits_at_k']
        ),
        TestCase(
            [text_feature()],
            ['loss', 'token_accuracy', 'last_accuracy', 'edit_distance',
             'perplexity']
        )
    ]
)
def test_validation_metrics(test_case: TestCase, csv_filename: str):
    # setup test scenarios
    test_scenarios = []
    for output_feature in test_case.output_features:
        # single output feature capture feature specific metrics
        of_name = output_feature[COLUMN]
        for metric in test_case.validation_metrics:
            test_scenarios.append((of_name, metric))
            if len(test_case.output_features) == 1:
                # it shoudl work when there's only one output feature
                # and the metric applyys to the output feature type,
                # the output feature name should be replacing combined
                # and a warning should be printed about the substitution
                test_scenarios.append(('combined', metric))

    # add standard test for combined
    test_scenarios.append(('combined', 'loss'))

    # setup features for the test
    input_features = [numerical_feature(),
                      category_feature(),
                      binary_feature()]
    output_features = test_case.output_features

    # generate training data
    training_data = generate_data(
        input_features,
        output_features,
        filename=csv_filename
    )

    # loop through scenarios
    for validation_field, validation_metric in test_scenarios:
        # setup config
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'training': {
                'epochs': 3,
                'validation_field': validation_field,
                'validation_metric': validation_metric
            }
        }

        model = LudwigModel(config)
        model.train(
            dataset=training_data,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_log=True,
            skip_save_model=True,
            skip_save_processed_input=True,
            skip_save_progress=True
        )


# test for multiple output features
@pytest.mark.parametrize(
    'test_case',
    [
        TestCase(
            [numerical_feature(), numerical_feature()],
            []
        ),
        TestCase(
            [category_feature(), numerical_feature()],
            []
        )
    ]
)
def test_validation_metrics_mulitiple_output(test_case: TestCase,
                                             csv_filename: str):
    test_validation_metrics(test_case, csv_filename)


# negative test for invalid metric name
@pytest.mark.parametrize(
    'test_case',
    [
        TestCase(
            [numerical_feature()],
            ['invalid_metric']
        )
    ]
)
def test_validation_invalid_metric(test_case: TestCase,
                                   csv_filename: str):
    # this should generate ValueError Exception
    try:
        test_validation_metrics(test_case, csv_filename)
        raise RuntimeError(
            'test_validation_metrics() should have raised ValueError '
            'but did not.'
        )
    except ValueError:
        pass
