import pytest

import torch

from ludwig.features.numerical_feature import NumericalInputFeature, \
    NumericalOutputFeature
from tests.integration_tests.utils import numerical_feature, \
    setup_input_feature_test, setup_output_feature_test

BATCH_SIZE = 2
FC_SIZE = 64
HIDDEN_SIZE = 16


@pytest.mark.parametrize('enc_encoder', ['passthrough', 'dense'])
def test_numerical_input_feature(enc_encoder: str) -> None:
    feature_to_test = numerical_feature(encoder=enc_encoder, fc_size=FC_SIZE)

    # setup synthetic tensor and feature definition
    input_tensor, feature_definition = setup_input_feature_test(
        batch_size=BATCH_SIZE,
        feature_definition=feature_to_test,
        feature_class=NumericalInputFeature
    )

    # instantiate binary input feature object
    feature_obj = NumericalInputFeature(feature_definition)

    # pass synthetic binary tensor through the input feature
    encoder_output = feature_obj(input_tensor)

    # confirm correctness of the the binary encoder output
    assert isinstance(encoder_output, dict)
    assert 'encoder_output' in encoder_output
    assert isinstance(encoder_output['encoder_output'], torch.Tensor)
    if enc_encoder == 'passthrough':
        assert encoder_output['encoder_output'].shape \
               == (BATCH_SIZE, 1)
    else:
        assert encoder_output['encoder_output'].shape \
               == (BATCH_SIZE, FC_SIZE)


@pytest.mark.parametrize(
    'loss_type',
    ['mean_squared_error', 'mean_absolute_error']
)
@pytest.mark.parametrize('num_fc_layers', [0, 1])
def test_numerical_output_feature(
        num_fc_layers: int,
        loss_type: str
) -> None:
    feature_to_test = numerical_feature(
        num_fc_layers=num_fc_layers,
        fc_size=FC_SIZE,
        loss={'type': loss_type}
    )

    # setup synthetic tensor and feature definition
    # ignore generated synthentic data
    input_for_output_feature, feature_definition = setup_output_feature_test(
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        feature_definition=feature_to_test,
        feature_class=NumericalOutputFeature
    )

    # instantiate binary output feature object
    feature_obj = NumericalOutputFeature(feature_definition)

    # pass synthetic binary tensor through the input feature
    # specify no dependencies
    feature_output = feature_obj(input_for_output_feature)

    # confirm correctness of the the binary encoder output structure
    assert isinstance(feature_output, dict)

    # confirm expected components of feature output
    assert 'logits' in feature_output
    assert feature_output['logits'].shape == (BATCH_SIZE,)

    assert 'last_hidden' in feature_output
    if num_fc_layers == 0:
        assert feature_output['last_hidden'].shape \
               == (BATCH_SIZE, HIDDEN_SIZE)
    else:
        assert feature_output['last_hidden'].shape \
               == (BATCH_SIZE, FC_SIZE)
