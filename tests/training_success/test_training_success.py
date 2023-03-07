import logging
from collections import deque
from pprint import pprint
from typing import Tuple

import pytest
import yaml
from configs import (
    combiner_type_to_combine_config_fn,
    ecd_config_section_to_config,
    feature_type_to_config_for_decoder_loss,
    feature_type_to_config_for_encoder_preprocessing,
)
from explore_schema import combine_configs, explore_properties

from ludwig.api import LudwigModel
from ludwig.config_validation.validation import get_schema
from ludwig.datasets import get_dataset
from ludwig.types import ModelConfigDict


def defaults_config_generator(feature_type, only_include) -> Tuple[ModelConfigDict, str]:
    assert isinstance(only_include, str)
    assert only_include in {"preprocessing", "encoder", "decoder", "loss"}

    schema = get_schema()
    properties = schema["properties"]["defaults"]["properties"][feature_type]["properties"]
    raw_entry = deque([(dict(), False)])
    explored = explore_properties(
        properties, parent_key="defaults." + feature_type, dq=raw_entry, only_include=[only_include]
    )

    if only_include in ["preprocessing", "encoder"]:
        config, dataset_name = feature_type_to_config_for_encoder_preprocessing[feature_type]
        config = yaml.safe_load(config)
    else:
        config, dataset_name = feature_type_to_config_for_decoder_loss[feature_type]
        config = yaml.safe_load(config)

    main_config_keys = list(config.keys())
    for key in main_config_keys:
        if key not in ["input_features", "output_features"]:
            del config[key]

    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}
    for config, dataset_name in combine_configs(explored, config, dataset_name):
        yield config, dataset_name


def ecd_trainer_config_generator() -> Tuple[ModelConfigDict, str]:
    schema = get_schema()
    properties = schema["properties"]

    raw_entry = deque([(dict(), False)])
    explored = explore_properties(properties, parent_key="", dq=raw_entry, only_include=["trainer"])
    config, dataset_name = ecd_config_section_to_config["trainer"]
    config = yaml.safe_load(config)
    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}

    for config, dataset_name in combine_configs(explored, config, dataset_name):
        yield config, dataset_name


def combiner_config_generator(combiner_type: str) -> Tuple[ModelConfigDict, str]:
    schema = get_schema()
    properties = schema["properties"]

    raw_entry = deque([(dict(), False)])
    explored = explore_properties(properties, parent_key="", dq=raw_entry, only_include=["combiner"])
    config, dataset_name = ecd_config_section_to_config[combiner_type]
    config = yaml.safe_load(config)
    config["model_type"] = "ecd"
    config["trainer"] = {"train_steps": 1}

    combine_configs_fn = combiner_type_to_combine_config_fn[combiner_type]
    for config, dataset_name in combine_configs_fn(explored, config, dataset_name):
        if config["combiner"]["type"] == combiner_type:
            yield config, dataset_name


def train_and_evaluate(config, dataset_name):
    # adding print statements to be captured in pytest stdout and help debug tests.
    print("Dataset name:", dataset_name)
    print("Config used")
    pprint(config)
    dataset_module = get_dataset(dataset_name)
    dataset = dataset_module.load()
    model = LudwigModel(config=config, callbacks=None, logging_level=logging.ERROR)
    model.train(dataset=dataset)
    model.evaluate(dataset=dataset)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("sequence_concat"))
def test_ecd_sequence_concat_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("sequence"))
def test_ecd_sequence_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("comparator"))
def test_ecd_comparator_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("concat"))
def test_ecd_concat_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("project_aggregate"))
def test_ecd_project_aggregate_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("tabnet"))
def test_ecd_tabnet_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("tabtransformer"))
def test_ecd_tabtransformer_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.combiner
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", combiner_config_generator("transformer"))
def test_ecd_transformer_combiner(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.ecd_trainer
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", ecd_trainer_config_generator())
def test_ecd_trainer(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "encoder"))
def test_number_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "decoder"))
def test_number_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "loss"))
def test_number_encoder_loss(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "preprocessing"))
def test_number_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "encoder"))
def test_category_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "decoder"))
def test_category_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "loss"))
def test_category_loss_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "preprocessing"))
def test_category_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "encoder"))
def test_binary_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "decoder"))
def test_binary_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "loss"))
def test_binary_loss_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.combinatorial
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "preprocessing"))
def test_binary_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


# @pytest.mark.text_feature
# @pytest.mark.combinatorial
# @pytest.mark.parametrize("config,dataset_name", defaults_config_generator("text", "preprocessing"))
# def test_text_preprocessing_defaults(config, dataset_name):
#     train_and_evaluate(config, dataset_name)


# @pytest.mark.text_feature
# @pytest.mark.combinatorial
# @pytest.mark.parametrize("config,dataset_name", defaults_config_generator("text", "encoder"))
# def test_text_encoder_defaults(config):
#     train_and_evaluate(config, dataset_name)
