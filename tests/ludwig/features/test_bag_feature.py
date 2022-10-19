from random import choice
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Dict

import pytest
import torch

from ludwig.constants import ENCODER
from ludwig.features.bag_feature import BagInputFeature
from ludwig.schema.features.bag_feature import BagInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
SEQ_SIZE = 20
BAG_W_SIZE = 256
EMBEDDING_SIZE = 5

CHARS = ascii_uppercase + ascii_lowercase + digits
VOCAB = ["".join(choice(CHARS) for _ in range(2)) for _ in range(256)]
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def bag_config():
    return {
        "name": "bag_feature",
        "type": "bag",
        "encoder": {
            "max_len": 5,
            "vocab_size": 10,
            "embedding_size": EMBEDDING_SIZE,
            "vocab": VOCAB,
        },
    }


@pytest.mark.parametrize("encoder", ["embed"])
def test_bag_input_feature(bag_config: Dict, encoder: str) -> None:
    bag_config[ENCODER].update({"type": encoder})
    bag_config, _ = load_config_with_kwargs(BagInputFeatureConfig, bag_config)
    bag_input_feature = BagInputFeature(bag_config).to(DEVICE)
    bag_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE, BAG_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = bag_input_feature(bag_tensor)
    assert encoder_output["encoder_output"].shape[1:][1:] == bag_input_feature.output_shape
