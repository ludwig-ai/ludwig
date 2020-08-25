import pytest
import tensorflow as tf

from ludwig.modules.reduction_modules import reduce_mode_registry
from tests.integration_tests.utils import generate_data, run_experiment
from tests.integration_tests.utils import sequence_feature, category_feature


@pytest.mark.parametrize('reduce_output', reduce_mode_registry)
def test_reduction(reduce_output, csv_filename):

    input_features = [
        sequence_feature(reduce_output=reduce_output)
    ]

    output_features = [
        category_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)


def test_dependencies_reduction(csv_filename):
    input_features = [
        sequence_feature(reduce_output=None)
    ]

    output_features = [
        category_feature(reduce_input=None, reduce_dependencies='sum'),
        category_feature(reduce_input=None, reduce_dependencies='sum')
    ]
    feature1_name = output_features[0]['name']
    output_features[1]['dependencies'] = [feature1_name]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)





