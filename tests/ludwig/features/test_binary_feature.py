import pytest

import torch

from ludwig.features.binary_feature import BinaryInputFeature
from tests.integration_tests.utils import binary_feature

BATCH_SIZE = 2
SEQ_SIZE = 20
DEFAULT_FC_SIZE = 256


@pytest.mark.parametrize(
    'enc_encoder',
    [
        'passthrough'
    ]
)
def test_binary_feature(enc_encoder):
    # synthetic binary tensor
    binary_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE],
                               dtype=torch.float32)

    # generate binary feature config
    binary_feature_config = binary_feature(
        folder='.',
        encoder=enc_encoder,
        max_sequence_length=SEQ_SIZE
    )

    # instantiate binary input feature object
    binary_input_feature = BinaryInputFeature(binary_feature_config)

    # pass synthetic binary tensor through the input feature
    encoder_output = binary_input_feature(binary_tensor)

    # confirm correctness of the the binary encoder output
    assert isinstance(encoder_output, dict)
    assert 'encoder_output' in encoder_output
    assert isinstance(encoder_output['encoder_output'], torch.Tensor)
    if enc_encoder == 'passthrough':
        assert encoder_output['encoder_output'].shape \
               == (BATCH_SIZE, 1, SEQ_SIZE)
    else:
        assert encoder_output['encoder_output'].shape \
               == (BATCH_SIZE, DEFAULT_FC_SIZE)
