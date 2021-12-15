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
    return {
        "name": "numerical_column_name",
        "type": "numerical",
        "vocab": ["a", "b", "c"],
        "representation": "dense",
        "encoder": "embed",
        "embedding_size": 50,
        "embeddings_trainable": True,
        "pretrained_embeddings": None,
        "embeddings_on_cpu": False,
        "fc_layers": None,
        "num_fc_layers": 0,
        "fc_size": 0,
        "use_bias": True,
        "weights_initializer": "uniform",
        "bias_initializer": "zeros",
        "weights_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "norm": None,
        "norm_params": None,
        "activation": "relu",
        "dropout": 0.0,
        "reduce_output": "sum",
        "tied_weights": None,
    }


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
    input_tensor = torch.randint(0, 3, size=(BATCH_SIZE,), dtype=torch.int32).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)