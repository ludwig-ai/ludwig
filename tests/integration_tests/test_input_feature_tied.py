from collections import namedtuple

import pytest

from ludwig.models.ecd import build_inputs
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import run_experiment
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import text_feature

# InputFeatureOptions namedtuple structure:
# feature_type: input feature type, e.g., numerical, category, etc.
# feature_options: None or dictionary of required input feature specification
# tie_features: boolean, True to tie features, False not to tie features
InputFeatureOptions = namedtuple('InputFeatureOptions',
                                 'feature_type feature_options tie_features')


# micro level test confirms the encoders for tied input features are sharing
# the same encoder.  Include negative tests to confirm untied input features
# do not share the same encoder.
# note: vocab parameter, below, is made up to facilitate creating input encoders
@pytest.mark.parametrize(
    'input_feature_options',
    [
        # tie input features, encoders should be the same
        InputFeatureOptions('numerical', None, True),
        InputFeatureOptions(
            'numerical',
            {'preprocessing': {'normalization': 'zscore'}},
            True
        ),
        InputFeatureOptions('binary', None, True),
        InputFeatureOptions('category', {'vocab': ['a', 'b', 'c']}, True),
        InputFeatureOptions('set', {'vocab': ['a', 'b', 'c']}, True),
        InputFeatureOptions('sequence', {'vocab': ['x', 'y', 'z']}, True),
        InputFeatureOptions('text', {'vocab': ['a', 'b', 'c']}, True),
        InputFeatureOptions('timeseries', {'should_embed': False}, True),
        InputFeatureOptions(
            'audio',
            {'embedding_size': 64, 'max_sequence_length': 16,
             'should_embed': False},
            True
        ),

        # do not tie input features, encoders should be different
        InputFeatureOptions('numerical', None, False),
        InputFeatureOptions(
            'numerical',
            {'preprocessing': {'normalization': 'zscore'}},
            False
        ),
        InputFeatureOptions('binary', None, False),
        InputFeatureOptions('category', {'vocab': ['a', 'b', 'c']}, False),
        InputFeatureOptions('set', {'vocab': ['a', 'b', 'c']}, False),
        InputFeatureOptions('sequence', {'vocab': ['x', 'y', 'z']}, False),
        InputFeatureOptions('text', {'vocab': ['a', 'b', 'c']}, False),
        InputFeatureOptions('timeseries', {'should_embed': False}, False),
        InputFeatureOptions(
            'audio',
            {'embedding_size': 64, 'max_sequence_length': 16,
             'should_embed': False},
            False
        ),
    ]
)
def test_tied_micro_level(input_feature_options):
    # build input feature config
    input_feature_configs = []

    input_feature_configs.append({
        'name': 'input_feature_1',
        'type': input_feature_options.feature_type
    })
    if input_feature_options.feature_options is not None:
        input_feature_configs[0].update(
            input_feature_options.feature_options)

    input_feature_configs.append({
        'name': 'input_feature_2',
        'type': input_feature_options.feature_type
    })
    if input_feature_options.feature_options is not None:
        input_feature_configs[1].update(
            input_feature_options.feature_options)

    # add tied option to the second feature
    if input_feature_options.tie_features:
        input_feature_configs[1]['tied'] = 'input_feature_1'

    input_features = build_inputs(input_feature_configs)

    if input_feature_options.tie_features:
        # should be same encoder
        assert input_features['input_feature_1'].encoder_obj is \
               input_features['input_feature_2'].encoder_obj
    else:
        # no tied parameter, encoders should be different
        assert input_features['input_feature_1'].encoder_obj is not \
               input_features['input_feature_2'].encoder_obj


# TiedUseCase namedtuple structure:
# input_feature: Ludwig synthetic data creation function.
# output_feature: Ludwig synthetic data creation function
TiedUseCase = namedtuple('TiedUseCase', 'input_feature output_feature')


# Macro level test ensures no exceptions are raised during a full_experiment()
@pytest.mark.parametrize(
    'tied_use_case',
    [
        TiedUseCase(numerical_feature, numerical_feature),
        TiedUseCase(text_feature, category_feature),
        TiedUseCase(sequence_feature, sequence_feature)
    ]
)
def test_tied_macro_level(tied_use_case: TiedUseCase, csv_filename: str):
    input_features = [
        numerical_feature(),  # Other feature
        tied_use_case.input_feature(),  # first feature to be tied
        tied_use_case.input_feature(),  # second feature to be tied
        category_feature()  # other feature
    ]
    # tie second feature to first feature
    input_features[2]['tied'] = input_features[1]['name']

    # setup output feature
    output_features = [
        tied_use_case.output_feature()
    ]

    # Generate test data and run full_experiment
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)
