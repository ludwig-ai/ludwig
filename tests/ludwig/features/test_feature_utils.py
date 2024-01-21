import numpy as np
import pytest
import torch

from ludwig.features import feature_utils


@pytest.fixture
def to_module() -> torch.nn.Module:
    return torch.nn.Module()


@pytest.fixture
def type_module() -> torch.nn.Module:
    return torch.nn.Module()


@pytest.fixture
def feature_dict(to_module: torch.nn.Module, type_module: torch.nn.Module) -> feature_utils.LudwigFeatureDict:
    fdict = feature_utils.LudwigFeatureDict()
    fdict.set("to", to_module)
    fdict.set("type", type_module)
    return fdict


def test_ludwig_feature_dict_get(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert feature_dict.get("to") == to_module
    assert feature_dict.get("type") == type_module


def test_ludwig_feature_dict_keys(feature_dict: feature_utils.LudwigFeatureDict):
    assert feature_dict.keys() == ["to", "type"]


def test_ludwig_feature_dict_values(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert list(feature_dict.values()) == [to_module, type_module]


def test_ludwig_feature_dict_items(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert feature_dict.items() == [("to", to_module), ("type", type_module)]


def test_ludwig_feature_dict_iter(feature_dict: feature_utils.LudwigFeatureDict):
    assert list(iter(feature_dict)) == ["to", "type"]
    assert list(feature_dict) == ["to", "type"]


def test_ludwig_feature_dict_len(feature_dict: feature_utils.LudwigFeatureDict):
    assert len(feature_dict) == 2


def test_ludwig_feature_dict_update(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    feature_dict.update({"to": torch.nn.Module(), "new": torch.nn.Module()})
    assert len(feature_dict) == 3
    assert not feature_dict.get("to") == to_module
    assert feature_dict.get("type") == type_module


def test_ludwig_feature_dict():
    feature_dict = feature_utils.LudwigFeatureDict()

    to_module = torch.nn.Module()
    type_module = torch.nn.Module()

    feature_dict.set("to", to_module)
    feature_dict.set("type", type_module)

    assert iter(feature_dict) is not None
    # assert next(feature_dict) is not None
    assert len(feature_dict) == 2
    assert feature_dict.keys() == ["to", "type"]
    assert feature_dict.items() == [("to", to_module), ("type", type_module)]
    assert feature_dict.get("to"), to_module

    feature_dict.update({"to_empty": torch.nn.Module()})

    assert len(feature_dict) == 3
    assert [key for key in feature_dict] == ["to", "type", "to_empty"]


def test_ludwig_feature_dict_with_periods():
    feature_dict = feature_utils.LudwigFeatureDict()

    to_module = torch.nn.Module()

    feature_dict.set("to.", to_module)

    assert feature_dict.keys() == ["to."]
    assert feature_dict.items() == [("to.", to_module)]
    assert feature_dict.get("to.") == to_module


@pytest.mark.parametrize("sequence_type", [list, tuple, np.array])
def test_compute_token_probabilities(sequence_type):
    inputs = sequence_type(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.6, 0.3, 0.2],
        ]
    )

    token_probabilities = feature_utils.compute_token_probabilities(inputs)
    assert np.allclose(token_probabilities, [0.7, 0.4, 0.6])


def test_compute_sequence_probability():
    inputs = np.array([0.7, 0.4, 0.6])

    sequence_probability = feature_utils.compute_sequence_probability(
        inputs, max_sequence_length=2, return_log_prob=False
    )

    assert np.allclose(sequence_probability, [0.28])  # 0.7 * 0.4
