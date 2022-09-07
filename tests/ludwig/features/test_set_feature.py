from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.constants import ENCODER
from ludwig.features.set_feature import SetInputFeature
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def set_config():
    return {
        "name": "set_column_name",
        "type": "set",
        "tied": None,
        "encoder": {
            "type": "embed",
            "vocab": ["a", "b", "c"],
            "representation": "dense",
            "embedding_size": 50,
            "embeddings_trainable": True,
            "pretrained_embeddings": None,
            "embeddings_on_cpu": False,
            "fc_layers": None,
            "num_fc_layers": 0,
            "use_bias": True,
            "weights_initializer": "uniform",
            "bias_initializer": "zeros",
            "norm": None,
            "norm_params": None,
            "activation": "relu",
            "dropout": 0.0,
            "reduce_output": "sum",
        },
    }


def test_set_input_feature(set_config: Dict) -> None:
    # setup image input feature definition
    set_def = deepcopy(set_config)

    # pickup any other missing parameters
    SetInputFeature.populate_defaults(set_def)

    # ensure no exceptions raised during build
    input_feature_obj = SetInputFeature(set_def).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = torch.randint(0, 2, size=(BATCH_SIZE, len(set_def[ENCODER]["vocab"])), dtype=torch.int64).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)
