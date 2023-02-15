import copy
import logging
from collections import deque

import pytest
import yaml
from configs import feature_type_to_config_for_decoder_loss, feature_type_to_config_for_encoder_preprocessing
from explore_schema import create_nested_dict, explore_properties, generate_possible_configs

from ludwig.api import LudwigModel
from ludwig.config_validation.validation import get_schema
from ludwig.datasets import get_dataset
from ludwig.utils.misc_utils import merge_dict


def defaults_config_generator(feature_type, only_include):
    assert isinstance(only_include, str)
    assert only_include in ["preprocessing", "encoder", "decoder", "loss"]

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
    config["trainer"] = {"train_steps": 2}

    for item in explored:
        for default_config in generate_possible_configs(config_options=item[0]):
            default_config = create_nested_dict(default_config)
            config = merge_dict(copy.deepcopy(config), default_config)
            yield config, dataset_name


def train_and_evaluate(config, dataset_name):
    dataset_module = get_dataset(dataset_name)
    dataset = dataset_module.load()
    model = LudwigModel(config=config, callbacks=None, logging_level=logging.ERROR)
    model.train(dataset=dataset)
    model.evaluate(dataset=dataset)


@pytest.mark.number_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "encoder"))
def test_number_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "decoder"))
def test_number_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "loss"))
def test_number_encoder_loss(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.number_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("number", "preprocessing"))
def test_number_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "encoder"))
def test_category_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "decoder"))
def test_category_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "loss"))
def test_category_loss_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.category_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("category", "preprocessing"))
def test_category_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "encoder"))
def test_binary_encoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "decoder"))
def test_binary_decoder_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "loss"))
def test_binary_loss_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.binary_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("binary", "preprocessing"))
def test_binary_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


@pytest.mark.text_feature
@pytest.mark.parametrize("config,dataset_name", defaults_config_generator("text", "preprocessing"))
def test_text_preprocessing_defaults(config, dataset_name):
    train_and_evaluate(config, dataset_name)


# @pytest.mark.text_feature
# @pytest.mark.parametrize("config,dataset_name", defaults_config_generator("text", "encoder"))
# def test_text_encoder_defaults(config):
#     train_and_evaluate(config, dataset_name)
