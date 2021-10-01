import logging

import torch

from ludwig.encoders.date_encoders import DateEmbed, DateWave

logger = logging.getLogger(__name__)


def test_date_embed():
    date_embed = DateEmbed()
    inputs = torch.tensor([[2, 1, 2, 2, 1, 6, 2, 5, 4, 1, 5, 2, 5, 3, 6, 4, 1, 3, 2],
                           [6, 2, 5, 1, 2, 4, 6, 3, 3, 2, 3, 3, 3, 1, 1, 3, 4, 4, 6]],
                          dtype=torch.int32)
    outputs = date_embed(inputs)
    assert outputs['encoder_output'].size()[1:] == date_embed.output_shape


def test_date_wave():
    date_embed = DateWave()
    inputs = torch.tensor([[2, 1, 2, 2, 1, 6, 2, 5, 4, 1, 5, 2, 5, 3, 6, 4, 1, 3, 2],
                           [6, 2, 5, 1, 2, 4, 6, 3, 3, 2, 3, 3, 3, 1, 1, 3, 4, 4, 6]],
                          dtype=torch.int32)
    outputs = date_embed(inputs)
    assert outputs['encoder_output'].size()[1:] == date_embed.output_shape
