import logging

import torch

from ludwig.encoders.date_encoders import DateEmbed, DateWave
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919
DEVICE = get_torch_device()

logger = logging.getLogger(__name__)


def test_date_embed():
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    date_embed = DateEmbed().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(date_embed, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"


def test_date_wave():
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    date_embed = DateWave().to(DEVICE)
    inputs = torch.tensor(
        [[2022, 6, 25, 5, 176, 9, 30, 59, 34259], [2022, 6, 25, 5, 176, 9, 30, 59, 34259]], dtype=torch.int32
    ).to(DEVICE)
    outputs = date_embed(inputs)
    assert outputs["encoder_output"].size()[1:] == date_embed.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(date_embed, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"
