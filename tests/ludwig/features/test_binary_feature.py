from typing import Dict

import pandas as pd
import pytest
import torch

from ludwig.features.binary_feature import BinaryFeatureMixin, BinaryInputFeature
from tests.integration_tests.utils import LocalTestBackend

SEQ_SIZE = 2
BINARY_W_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def binary_config():
    return {
        "name": "binary_feature",
        "type": "binary",
    }


@pytest.mark.parametrize("encoder", ["passthrough"])
def test_binary_input_feature(binary_config: Dict, encoder: str) -> None:
    binary_config.update({"encoder": encoder})
    binary_input_feature = BinaryInputFeature(binary_config)
    binary_tensor = torch.randn([SEQ_SIZE, BINARY_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = binary_input_feature(binary_tensor)
    assert encoder_output["encoder_output"].shape[1:] == binary_input_feature.output_shape


def test_add_feature_data():
    feature_meta = BinaryFeatureMixin.get_feature_meta(pd.Series(["0", "T", "F"]), {}, LocalTestBackend())

    assert feature_meta["str2bool"] == {"0": False, "T": True, "F": False}
    assert feature_meta["bool2str"] == ["0", "F", "T"]
    assert feature_meta["fallback_true_label"] is None

    feature_meta = BinaryFeatureMixin.get_feature_meta(pd.Series(["1", "T", "F"]), {}, LocalTestBackend())

    assert feature_meta["str2bool"] == {"1": True, "T": True, "F": False}
    assert feature_meta["bool2str"] == ["F", "1", "T"]
    assert feature_meta["fallback_true_label"] is None
