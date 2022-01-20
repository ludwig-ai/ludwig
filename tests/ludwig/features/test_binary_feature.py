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


def test_get_feature_meta_default_fill_false():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["0", "T", "F"]), {"fill_value": "0"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"0": False, "T": True, "F": False}
    assert feature_meta["bool2str"] == ["F", "T"]
    assert feature_meta["fallback_true_label"] is None


def test_get_feature_meta_default_fill_true():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["1", "T", "F"]), {"fill_value": "1"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"1": True, "T": True, "F": False}
    assert feature_meta["bool2str"] == ["F", "T"]
    assert feature_meta["fallback_true_label"] is None


def test_get_feature_meta_default_fill_unused():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["T", "F"]), {"fill_value": "False"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"T": True, "F": False}
    assert feature_meta["bool2str"] == ["F", "T"]
    assert feature_meta["fallback_true_label"] is None


def test_get_feature_meta_unconventional_bool_default_fill():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["human", "bot", "False"]), {"fill_value": "False"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"human": False, "bot": True, "False": False}
    assert feature_meta["bool2str"] == ["human", "bot"]
    assert feature_meta["fallback_true_label"] == "bot"


def test_get_feature_meta_conventional_bool_mix():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["human", "False"]), {"fill_value": "False"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"human": True, "False": False}
    assert feature_meta["bool2str"] == ["False", "human"]
    assert feature_meta["fallback_true_label"] == "human"


def test_get_feature_meta_unconventional_bool_default_fill_with_fallback_true_label():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["human", "bot", "False"]),
        {"fill_value": "False", "fallback_true_label": "human"},
        LocalTestBackend(),
    )
    assert feature_meta["str2bool"] == {"human": True, "bot": False, "False": False}
    assert feature_meta["bool2str"] == ["bot", "human"]
    assert feature_meta["fallback_true_label"] == "human"


def test_get_feature_meta_unconventional_bool_default_fill_unused():
    feature_meta = BinaryFeatureMixin.get_feature_meta(
        pd.Series(["human", "bot"]), {"fill_value": "False"}, LocalTestBackend()
    )
    assert feature_meta["str2bool"] == {"human": False, "bot": True}
    assert feature_meta["bool2str"] == ["human", "bot"]
    assert feature_meta["fallback_true_label"] == "bot"


def test_get_feature_meta_too_many_values():
    with pytest.raises(Exception):
        BinaryFeatureMixin.get_feature_meta(
            pd.Series(["human", "False", "T", "F"]), {"fill_value": "False"}, LocalTestBackend()
        )

    with pytest.raises(Exception):
        BinaryFeatureMixin.get_feature_meta(pd.Series(["0", "T", "F"]), {"fill_value": "False"}, LocalTestBackend())
