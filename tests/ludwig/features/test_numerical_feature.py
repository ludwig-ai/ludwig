from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.features.numerical_feature import NumericalInputFeature
from ludwig.models.ecd import build_single_input

BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def numerical_config():
    return {"name": "numerical_column_name", "type": "numerical"}


def test_numerical_input_feature(
    numerical_config: Dict,
) -> None:
    # setup image input feature definition
    numerical_def = deepcopy(numerical_config)

    # pickup any other missing parameters
    NumericalInputFeature.populate_defaults(numerical_def)

    # ensure no exceptions raised during build
    input_feature_obj = build_single_input(numerical_def, None).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = torch.rand(2, dtype=torch.float32).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)
