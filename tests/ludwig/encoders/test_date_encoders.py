import logging

import torch

from ludwig.encoders.date_encoders import DateEmbed, DateWave

logger = logging.getLogger(__name__)


def test_date_embed():
    date_embed = DateEmbed()
    inputs = torch.tensor([[2022, 6, 25, 5, 176, 9, 30, 59, 34259],
                           [2022, 6, 25, 5, 176, 9, 30, 59, 34259]],
                          dtype=torch.int32)
    outputs = date_embed(inputs)
    assert outputs['encoder_output'].size()[1:] == date_embed.output_shape


def test_date_wave():
    date_embed = DateWave()
    inputs = torch.tensor([[2022, 6, 25, 5, 176, 9, 30, 59, 34259],
                           [2022, 6, 25, 5, 176, 9, 30, 59, 34259]],
                          dtype=torch.int32)
    outputs = date_embed(inputs)
    assert outputs['encoder_output'].size()[1:] == date_embed.output_shape
