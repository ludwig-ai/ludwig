import numpy as np
import pytest
import torch

from ludwig.features import feature_utils


@pytest.fixture
def to_module() -> torch.nn.Module:
    """Dummy Module to test the LudwigFeatureDict."""
    return torch.nn.Module()


@pytest.fixture
def type_module() -> torch.nn.Module:
    """Dummy Module to test the LudwigFeatureDict."""
    return torch.nn.Module()


@pytest.fixture
def feature_dict(to_module: torch.nn.Module, type_module: torch.nn.Module) -> feature_utils.LudwigFeatureDict:
    fdict = feature_utils.LudwigFeatureDict()
    fdict.set("to", to_module)
    fdict["type"] = type_module
    return fdict


def test_ludwig_feature_dict_get(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert feature_dict["to"] == to_module
    assert feature_dict.get("type") == type_module
    assert feature_dict.get("other_key", default=None) is None


def test_ludwig_feature_dict_keys(feature_dict: feature_utils.LudwigFeatureDict):
    assert list(feature_dict.keys()) == ["to", "type"]


def test_ludwig_feature_dict_values(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert list(feature_dict.values()) == [to_module, type_module]


def test_ludwig_feature_dict_items(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    assert list(feature_dict.items()) == [("to", to_module), ("type", type_module)]


def test_ludwig_feature_dict_iter(feature_dict: feature_utils.LudwigFeatureDict):
    assert list(iter(feature_dict)) == ["to", "type"]
    assert list(feature_dict) == ["to", "type"]


def test_ludwig_feature_dict_len(feature_dict: feature_utils.LudwigFeatureDict):
    assert len(feature_dict) == 2


def test_ludwig_feature_dict_contains(feature_dict: feature_utils.LudwigFeatureDict):
    assert "to" in feature_dict and "type" in feature_dict


def test_ludwig_feature_dict_eq(feature_dict: feature_utils.LudwigFeatureDict):
    other_dict = feature_utils.LudwigFeatureDict()
    assert not feature_dict == other_dict
    other_dict.update(feature_dict.items())
    assert feature_dict == other_dict


def test_ludwig_feature_dict_update(
    feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module, type_module: torch.nn.Module
):
    feature_dict.update({"to": torch.nn.Module(), "new": torch.nn.Module()})
    assert len(feature_dict) == 3
    assert not feature_dict.get("to") == to_module
    assert feature_dict.get("type") == type_module


def test_ludwig_feature_dict_del(feature_dict: feature_utils.LudwigFeatureDict):
    del feature_dict["to"]
    assert len(feature_dict) == 1


def test_ludwig_feature_dict_clear(feature_dict: feature_utils.LudwigFeatureDict):
    feature_dict.clear()
    assert len(feature_dict) == 0


def test_ludwig_feature_dict_pop(feature_dict: feature_utils.LudwigFeatureDict, type_module: torch.nn.Module):
    assert feature_dict.pop("type") == type_module
    assert len(feature_dict) == 1
    assert feature_dict.pop("type", default=None) is None


def test_ludwig_feature_dict_popitem(feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module):
    assert feature_dict.popitem() == ("to", to_module)
    assert len(feature_dict) == 1


def test_ludwig_feature_dict_setdefault(feature_dict: feature_utils.LudwigFeatureDict, to_module: torch.nn.Module):
    assert feature_dict.setdefault("to") == to_module
    assert feature_dict.get("other_key") is None


@pytest.mark.parametrize("name", ["to", "type", "foo", "foo.bar"])
def test_name_to_module_dict_key(name: str):
    key = feature_utils.get_module_dict_key_from_name(name)
    assert key != name
    assert "." not in key
    assert feature_utils.get_name_from_module_dict_key(key) == name


def test_ludwig_feature_dict():
    feature_dict = feature_utils.LudwigFeatureDict()

    to_module = torch.nn.Module()
    type_module = torch.nn.Module()

    feature_dict.set("to", to_module)
    feature_dict.set("type", type_module)

    assert iter(feature_dict) is not None
    assert len(feature_dict) == 2
    assert list(feature_dict.keys()) == ["to", "type"]
    assert list(feature_dict.items()) == [("to", to_module), ("type", type_module)]
    assert feature_dict.get("to"), to_module

    feature_dict.update({"to_empty": torch.nn.Module()})

    assert len(feature_dict) == 3
    assert [key for key in feature_dict] == ["to", "type", "to_empty"]


def test_ludwig_feature_dict_with_periods():
    feature_dict = feature_utils.LudwigFeatureDict()

    to_module = torch.nn.Module()

    feature_dict.set("to.", to_module)

    assert list(feature_dict.keys()) == ["to."]
    assert list(feature_dict.items()) == [("to.", to_module)]
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
