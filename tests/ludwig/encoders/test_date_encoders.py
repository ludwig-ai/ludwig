import logging

import torch

from ludwig.encoders.date_encoders import DateEmbed, DateWave

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_date_embed():
    date_embed = DateEmbed().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape()


def test_date_wave():
    date_embed = DateWave().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape()
