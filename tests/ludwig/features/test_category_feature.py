from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.constants import ENCODER, TYPE
from ludwig.features.category_feature import CategoryInputFeature
from ludwig.models.ecd import ECD
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def category_config():
    return {
        "name": "category_column_name",
        "type": "category",
        "tied": None,
        "encoder": {
            "embedding_size": 256,
            "embeddings_on_cpu": False,
            "pretrained_embeddings": None,
            "embeddings_trainable": True,
            "dropout": 0.0,
            "vocab": ["a", "b", "c"],
            "embedding_initializer": None,
        },
    }


@pytest.mark.parametrize("encoder", ["dense", "sparse"])
def test_category_input_feature(
    category_config: Dict,
    encoder: str,
) -> None:
    # setup image input feature definition
    category_def = deepcopy(category_config)
    category_def[ENCODER][TYPE] = encoder

    # pickup any other missing parameters
    CategoryInputFeature.populate_defaults(category_def)

    # ensure no exceptions raised during build
    input_feature_obj = ECD.build_single_input(category_def, None)

    # check one forward pass through input feature
    input_tensor = torch.randint(0, 3, size=(BATCH_SIZE,), dtype=torch.int32).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)
