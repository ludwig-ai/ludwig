from collections import namedtuple

import pytest

from ludwig.models.ecd import build_inputs


# InputFeatureOptions namedtuple structure:
# feature_type: input feature type, e.g., numerical, category, etc.
# feature_options: None or dictionary of required input feature specifications
#   dictionary can be empty, None indicates not to tie the two input features
InputFeatureOptions = namedtuple('InputFeatureOptions', 'feature_type feature_options')


# note: vocab parameter, below, is made up to facilitate creating input encoders
@pytest.mark.parametrize(
    'input_feature_options',
    [
        InputFeatureOptions('numerical', {}),
        InputFeatureOptions(
            'numerical',
            {'preprocessing': {'normalization': 'zscore'}}
        ),
        InputFeatureOptions('binary', {}),
        InputFeatureOptions('category', {'vocab': ['a', 'b', 'c']}),
        InputFeatureOptions('set', {'vocab': ['a', 'b', 'c']}),
        InputFeatureOptions('sequence', {'vocab': ['x', 'y', 'z']}),
        InputFeatureOptions('text', {'vocab': ['a', 'b', 'c']}),
        InputFeatureOptions('timeseries', {'vocab': ['a', 'b', 'c']}),
        InputFeatureOptions(
            'audio',
            {'embedding_size': 64, 'length': 16, 'vocab': ['a', 'b', 'c']}
        ),

        # do not tie input features, encoders should be different
        InputFeatureOptions('numerical', None)
    ]
)
def test_encoder_tied_weights(input_feature_options):

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
    if input_feature_options.feature_options is not None:
        input_feature_definitions[1]['tied'] = 'input_feature_1'

    input_features = build_inputs(input_feature_definitions)

    if input_feature_options.feature_options is not None:
        # should be same encoder
        assert input_features['input_feature_1'].encoder_obj is \
            input_features['input_feature_2'].encoder_obj
    else:
        # no tied parameter, encoders should be different
        assert input_features['input_feature_1'].encoder_obj is not \
               input_features['input_feature_2'].encoder_obj









