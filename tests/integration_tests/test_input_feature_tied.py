from collections import namedtuple

import pytest

from ludwig.models.ecd import build_inputs


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
        InputFeatureOptions('timeseries', {'vocab': ['a', 'b', 'c']}, True),
        InputFeatureOptions(
            'audio',
            {'embedding_size': 64, 'length': 16, 'vocab': ['a', 'b', 'c']},
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
        InputFeatureOptions('timeseries', {'vocab': ['a', 'b', 'c']}, False),
        InputFeatureOptions(
            'audio',
            {'embedding_size': 64, 'length': 16, 'vocab': ['a', 'b', 'c']},
            False
        ),
    ]
)
def test_tied_micro_level(input_feature_options):

    # build input feature definition
    input_feature_definitions = []

    input_feature_definitions.append({
        'name': 'input_feature_1',
        'type': input_feature_options.feature_type
    })
    if input_feature_options.feature_options is not None:
        input_feature_definitions[0].update(input_feature_options.feature_options)

    input_feature_definitions.append({
        'name': 'input_feature_2',
        'type': input_feature_options.feature_type
    })
    if input_feature_options.feature_options is not None:
        input_feature_definitions[1].update(input_feature_options.feature_options)

    # add tied option to the second feature
    if input_feature_options.tie_features:
        input_feature_definitions[1]['tied'] = 'input_feature_1'

    input_features = build_inputs(input_feature_definitions)

    if input_feature_options.tie_features:
        # should be same encoder
        assert input_features['input_feature_1'].encoder_obj is \
            input_features['input_feature_2'].encoder_obj
    else:
        # no tied parameter, encoders should be different
        assert input_features['input_feature_1'].encoder_obj is not \
               input_features['input_feature_2'].encoder_obj


# Macro level test ensures no exceptions are raised during a full_experiment()








