from typing import Tuple, List, Union
import pytest

import numpy as np

import torch

from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.text_feature import TextInputFeature
from tests.integration_tests.utils import sequence_feature, \
    setup_input_feature_test
from tests.integration_tests.utils import ENCODERS

BATCH_SIZE = 2  # 8
SEQ_SIZE = 6
VOCAB_SIZE = 64


@pytest.mark.parametrize('encoder', ENCODERS)
@pytest.mark.parametrize(
    'sequence_type',
    [SequenceInputFeature, TextInputFeature]
)
def test_sequence_input_feature(
        encoder: str,
        sequence_type: Union[SequenceInputFeature]
) -> None:
    feature_to_test = sequence_feature(
        vocab_size=VOCAB_SIZE,
        min_len=2,
        max_len=SEQ_SIZE,
        encoder='rnn',
    )

    # setup synthetic tensor and feature definition
    input_tensor, feature_definition = setup_input_feature_test(
        batch_size=BATCH_SIZE,
        feature_definition=feature_to_test,
        feature_class=SequenceInputFeature
    )

    # create sequence input feature object
    input_feature_obj = sequence_type(feature_definition)

    # confirm dtype property
    assert input_feature_obj.input_dtype == torch.int32

    # todo: confirm how to check for input_shape when all sequences
    #   are not maximum length
    # assert input_feature_obj.input_shape == (SEQ_SIZE,)

    # do one forward pass through input feature/encoder
    encoder_output = input_feature_obj(input_tensor)

    # confirm encoder output has required parts
    assert 'encoder_output' in encoder_output
    assert 'encoder_output_state' in encoder_output
    assert 'lengths' in encoder_output

    # confirm encoder output shape
    assert encoder_output['encoder_output'].shape == \
           (BATCH_SIZE, *input_feature_obj.output_shape)


# todo: add unit test for sequence output feature
def test_sequence_output_feature():
    pass
