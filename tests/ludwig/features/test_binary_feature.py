from typing import Dict

import pytest
import torch

from ludwig.constants import ENCODER, ENCODER_OUTPUT
from ludwig.features.binary_feature import BinaryInputFeature, BinaryOutputFeature
from ludwig.schema.features.binary_feature import BinaryInputFeatureConfig, BinaryOutputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
BINARY_W_SIZE = 1
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def binary_config():
    return {
        "name": "binary_feature",
        "type": "binary",
    }


@pytest.mark.parametrize("encoder", ["passthrough", "dense"])
def test_binary_input_feature(binary_config: Dict, encoder: str):
    binary_config.update({ENCODER: {"type": encoder}})
    binary_config, _ = load_config_with_kwargs(BinaryInputFeatureConfig, binary_config)
    binary_input_feature = BinaryInputFeature(binary_config).to(DEVICE)

    binary_tensor = binary_input_feature.create_sample_input(batch_size=BATCH_SIZE)
    assert binary_tensor.shape == torch.Size([BATCH_SIZE])
    assert binary_tensor.dtype == torch.bool

    encoder_output = binary_input_feature(binary_tensor)

    assert encoder_output[ENCODER_OUTPUT].shape[1:] == binary_input_feature.output_shape


def test_binary_output_feature():
    binary_output_config = {
        "name": "binary_feature",
        "type": "binary",
        "input_size": BINARY_W_SIZE,
        "decoder": {
            "type": "regressor",
            "input_size": 1,
        },
        "loss": {
            "type": "binary_weighted_cross_entropy",
            "positive_class_weight": 1,
            "robust_lambda": 0,
            "confidence_penalty": 0,
        },
    }
    binary_output_config, _ = load_config_with_kwargs(BinaryOutputFeatureConfig, binary_output_config)
    binary_output_feature = BinaryOutputFeature(binary_output_config, {}).to(DEVICE)
    combiner_outputs = dict()
    combiner_outputs["combiner_output"] = torch.randn([BATCH_SIZE, BINARY_W_SIZE], dtype=torch.float32).to(DEVICE)

    binary_output = binary_output_feature(combiner_outputs, {})

    assert "last_hidden" in binary_output
    assert "logits" in binary_output
    assert binary_output["logits"].size() == torch.Size([BATCH_SIZE])


def test_binary_output_feature_without_positive_class_weight():
    binary_output_config = {
        "name": "binary_feature",
        "type": "binary",
        "input_size": BINARY_W_SIZE,
        "decoder": {
            "type": "regressor",
            "input_size": 1,
        },
        "loss": {
            "type": "binary_weighted_cross_entropy",
            "positive_class_weight": None,
            "robust_lambda": 0,
            "confidence_penalty": 0,
        },
    }
    binary_output_config, _ = load_config_with_kwargs(BinaryOutputFeatureConfig, binary_output_config)
    binary_output_feature = BinaryOutputFeature(binary_output_config, {}).to(DEVICE)
    combiner_outputs = {}
    combiner_outputs["combiner_output"] = torch.randn([BATCH_SIZE, BINARY_W_SIZE], dtype=torch.float32).to(DEVICE)

    binary_output = binary_output_feature(combiner_outputs, {})

    assert "last_hidden" in binary_output
    assert "logits" in binary_output
    assert binary_output["logits"].size() == torch.Size([BATCH_SIZE])
