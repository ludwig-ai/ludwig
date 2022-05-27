import logging

import torch

from ludwig.encoders.date_encoders import DateEmbed, DateWave
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)

DEVICE = get_torch_device()


def test_date_embed():
    date_embed = DateEmbed().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape


def test_date_wave():
    date_embed = DateWave().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape
