import pytest
import torch

from ludwig.constants import (ENCODER_OUTPUT, ENCODER_OUTPUT_STATE, HIDDEN,
                              LOGITS)
from ludwig.decoders.image_decoders import UNetDecoder
from ludwig.encoders.image.base import UNetEncoder
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import \
    check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("height,width,num_channels,num_classes", [(224, 224, 1, 2), (224, 224, 3, 8)])
@pytest.mark.parametrize("batch_size", [4, 1])
def test_unet_decoder(height, width, num_channels, num_classes, batch_size):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    unet_encoder = UNetEncoder(height=height, width=width, num_channels=num_channels)
    inputs = torch.rand(batch_size, num_channels, height, width)
    encoder_outputs = unet_encoder(inputs)
    assert encoder_outputs[ENCODER_OUTPUT].shape[1:] == unet_encoder.output_shape
    assert len(encoder_outputs[ENCODER_OUTPUT_STATE]) == 4

    hidden = torch.reshape(encoder_outputs[ENCODER_OUTPUT], [batch_size, -1])

    unet_decoder = UNetDecoder(hidden.size(dim=1), height, width, 1, num_classes)
    combiner_outputs = {
        HIDDEN: hidden,
        ENCODER_OUTPUT_STATE: encoder_outputs[ENCODER_OUTPUT_STATE].copy(),  # create a copy
    }

    output = unet_decoder(combiner_outputs, target=None)

    assert list(output[LOGITS].size()) == [batch_size, num_classes, height, width]

    # check for parameter updating
    target = torch.randn(output[LOGITS].shape)
    combiner_outputs[ENCODER_OUTPUT_STATE] = encoder_outputs[ENCODER_OUTPUT_STATE]  # restore state
    fpc, tpc, upc, not_updated = check_module_parameters_updated(unet_decoder, (combiner_outputs, None), target)
    assert upc == tpc, f"Failed to update parameters. Parameters not updated: {not_updated}"
