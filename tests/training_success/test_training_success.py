import logging
from collections import deque
from pprint import pprint
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
import yaml

from ludwig.api import LudwigModel
from ludwig.config_sampling.explore_schema import combine_configs, ConfigOption, explore_properties
from ludwig.config_validation.validation import get_schema
from ludwig.types import ModelConfigDict

from .configs import (
    COMBINER_TYPE_TO_COMBINE_FN_MAP,
    ECD_CONFIG_SECTION_TO_CONFIG,
    FEATURE_TYPE_TO_CONFIG_FOR_DECODER_LOSS,
    FEATURE_TYPE_TO_CONFIG_FOR_ENCODER_PREPROCESSING,
)


def defaults_config_generator(
    feature_type: str, allow_list: str, static_schema: Dict[str, Any] = None
) -> Tuple[ModelConfigDict, pd.DataFrame]:
    """Generate combinatorial configs for the defaults section of the Ludwig config.

    Args:
        feature_type: feature type to explore.
        allow_list: top-level parameter of the defaults sections that should be included.
    """
    assert isinstance(allow_list, str)
    assert allow_list in {"preprocessing", "encoder", "decoder", "loss"}

    schema = get_schema() if not static_schema else static_schema
    properties = schema["properties"]["defaults"]["properties"][feature_type]["properties"]
    raw_entry = deque([ConfigOption(dict(), False)])
    explored = explore_properties(
        properties, parent_parameter_path=f"defaults.{feature_type}", dq=raw_entry, allow_list=[allow_list]
    )

    if allow_list in ["preprocessing", "encoder"]:
        config = FEATURE_TYPE_TO_CONFIG_FOR_ENCODER_PREPROCESSING[feature_type]
        config = yaml.safe_load(config)
    else:  # decoder and loss
        config = FEATURE_TYPE_TO_CONFIG_FOR_DECODER_LOSS[feature_type]
        config = yaml.safe_load(config)

    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}

    combined_configs = combine_configs(explored, config)
    logging.info(f"Generated {len(combined_configs)} for {feature_type} {allow_list} combinatorial tests.")

    for config, dataset in combined_configs:
        yield config, dataset


def ecd_trainer_config_generator(static_schema: Dict[str, Any] = None) -> Tuple[ModelConfigDict, pd.DataFrame]:
    """Generate combinatorial configs for the ECD trainer section of the Ludwig config."""
    schema = get_schema() if not static_schema else static_schema
    properties = schema["properties"]

    raw_entry = deque([ConfigOption(dict(), False)])
    explored = explore_properties(properties, parent_parameter_path="", dq=raw_entry, allow_list=["trainer"])
    config = ECD_CONFIG_SECTION_TO_CONFIG["trainer"]
    config = yaml.safe_load(config)
    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}

    combined_configs = combine_configs(explored, config)

    # HACK(Arnav): Remove configs that have LARS, LAMB or Lion optimizers, or Paged or 8-bit optimizers.
    # This is because they require GPUs.
    filtered_configs = []

    for config, dataset in combined_configs:
        optimizer_type = config.get("trainer", {}).get("optimizer", "").get("type", "")

        if optimizer_type not in {"lars", "lamb", "lion"} and not (
            "paged" in optimizer_type or "8bit" in optimizer_type
        ):
            filtered_configs.append((config, dataset))

    # Replace combined_configs with the filtered_configs
    combined_configs = filtered_configs

    logging.info(f"Generated {len(combined_configs)} for ECD trainer combinatorial tests.")

    for config, dataset in combined_configs:
        yield config, dataset


def combiner_config_generator(
    combiner_type: str, static_schema: Dict[str, Any] = None
) -> Tuple[ModelConfigDict, pd.DataFrame]:
    """Generate combinatorial configs for the combiner section of the Ludwig config.

    Args:
        combiner_type: combiner type to explore.
    """
    schema = get_schema() if not static_schema else static_schema
    properties = schema["properties"]

    raw_entry = deque([ConfigOption(dict(), False)])
    explored = explore_properties(properties, parent_parameter_path="", dq=raw_entry, allow_list=["combiner"])
    config = ECD_CONFIG_SECTION_TO_CONFIG[combiner_type]
    config = yaml.safe_load(config)
    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}

    combine_configs_fn = COMBINER_TYPE_TO_COMBINE_FN_MAP[combiner_type]

    combined_configs = combine_configs_fn(explored, config)
    combined_configs = [
        (config, dataset) for config, dataset in combined_configs if config["combiner"]["type"] == combiner_type
    ]
    logging.info(f"Generated {len(combined_configs)} for {combiner_type} combiner combinatorial tests.")

    for config, dataset in combined_configs:
        yield config, dataset


def train_and_evaluate(config: ModelConfigDict, dataset: pd.DataFrame):
    """Trains and evaluates a model with the given config.

    Args:
        config: valid Ludwig config.
        dataset: Ludwig dataset name to train on.
    """
    # adding print statements to be captured in pytest stdout and help debug tests.
    print("Config used (trained on synthetic data)")
    pprint(config)
    model = LudwigModel(config=config, callbacks=None, logging_level=logging.ERROR)
    model.train(dataset=dataset)
    model.evaluate(dataset=dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("sequence_concat"))
def test_ecd_sequence_concat_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("sequence"))
def test_ecd_sequence_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("comparator"))
def test_ecd_comparator_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("concat"))
def test_ecd_concat_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("project_aggregate"))
def test_ecd_project_aggregate_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("tabnet"))
def test_ecd_tabnet_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("tabtransformer"))
def test_ecd_tabtransformer_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", combiner_config_generator("transformer"))
def test_ecd_transformer_combiner(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", ecd_trainer_config_generator())
def test_ecd_trainer(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("number", "encoder"))
def test_number_encoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("number", "decoder"))
def test_number_decoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("number", "loss"))
def test_number_encoder_loss(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("number", "preprocessing"))
def test_number_preprocessing_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("category", "encoder"))
def test_category_encoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("category", "decoder"))
def test_category_decoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("category", "loss"))
def test_category_loss_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("category", "preprocessing"))
def test_category_preprocessing_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("binary", "encoder"))
def test_binary_encoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("binary", "decoder"))
def test_binary_decoder_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("binary", "loss"))
def test_binary_loss_defaults(config, dataset):
    train_and_evaluate(config, dataset)


@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset", defaults_config_generator("binary", "preprocessing"))
def test_binary_preprocessing_defaults(config, dataset):
    train_and_evaluate(config, dataset)


# @pytest.mark.combinatorial
# @pytest.mark.parametrize("config,dataset", defaults_config_generator("text", "preprocessing"))
# def test_text_preprocessing_defaults(config, dataset):
#     train_and_evaluate(config, dataset)


# @pytest.mark.combinatorial
# @pytest.mark.parametrize("config,dataset", defaults_config_generator("text", "encoder"))
# def test_text_encoder_defaults(config):
#     train_and_evaluate(config, dataset)
