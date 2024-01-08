from typing import Any, Dict, Optional

import pytest

from ludwig.constants import (
    BASE_MODEL,
    BINARY,
    ENCODER,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_LLM,
    MODEL_TYPE,
    NAME,
    OUTPUT_FEATURES,
    TEXT,
    TYPE,
)
from ludwig.schema.encoders.text_encoders import BERTConfig
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.preprocessing.text import TextPreprocessingConfig
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.config_utils import is_or_uses_llm


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [None, "bert-large-uncased"],
    ids=["default_model", "override_model"],
)
def test_set_fixed_preprocessing_params(pretrained_model_name_or_path: str):
    expected_model_name = "bert-base-uncased"

    preprocessing = TextPreprocessingConfig.from_dict(
        {
            "tokenizer": "space",
            "lowercase": True,
        }
    )

    encoder_params = {}
    if pretrained_model_name_or_path is not None:
        encoder_params["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        expected_model_name = pretrained_model_name_or_path

    encoder = BERTConfig.from_dict(encoder_params)
    encoder.set_fixed_preprocessing_params(MODEL_ECD, preprocessing)

    assert preprocessing.tokenizer == "hf_tokenizer"
    assert preprocessing.lowercase
    assert preprocessing.pretrained_model_name_or_path == expected_model_name


@pytest.mark.parametrize(
    "encoder_params,expected",
    [
        ({"type": "parallel_cnn"}, False),
        ({"type": "bert", "trainable": False}, True),
        ({"type": "bert", "trainable": True}, False),
    ],
    ids=["parallel_cnn", "bert_fixed", "bert_trainable"],
)
def test_set_fixed_preprocessing_params_cache_embeddings(encoder_params: Dict[str, Any], expected: Optional[bool]):
    preprocessing = TextPreprocessingConfig.from_dict(
        {
            "tokenizer": "space",
            "lowercase": True,
            "cache_encoder_embeddings": True,
        }
    )

    encoder = get_encoder_cls(MODEL_ECD, TEXT, encoder_params[TYPE]).from_dict(encoder_params)
    encoder.set_fixed_preprocessing_params(MODEL_ECD, preprocessing)
    assert preprocessing.cache_encoder_embeddings == expected


@pytest.fixture()
def llm_config_dict():
    return {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1"}],
        OUTPUT_FEATURES: [{TYPE: TEXT, NAME: "out1"}],
    }


@pytest.fixture()
def llm_config_object(llm_config_dict):
    return ModelConfig.from_dict(llm_config_dict)


@pytest.fixture()
def ecd_config_dict_llm_encoder():
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [
            {
                TYPE: TEXT,
                NAME: "in1",
                ENCODER: {TYPE: MODEL_LLM, BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM"},
            }
        ],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture()
def ecd_config_object_llm_encoder(ecd_config_dict_llm_encoder):
    return ModelConfig.from_dict(ecd_config_dict_llm_encoder)


@pytest.fixture()
def ecd_config_dict_llm_encoder_multiple_features():
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [
            {TYPE: BINARY, NAME: "in1"},
            {
                TYPE: TEXT,
                NAME: "in2",
                ENCODER: {TYPE: MODEL_LLM, BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM"},
            },
        ],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture()
def ecd_config_object_llm_encoder_multiple_features(ecd_config_dict_llm_encoder_multiple_features):
    return ModelConfig.from_dict(ecd_config_dict_llm_encoder_multiple_features)


@pytest.fixture()
def ecd_config_dict_no_llm_encoder():
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1", ENCODER: {TYPE: "parallel_cnn"}}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture()
def ecd_config_object_no_llm_encoder(ecd_config_dict_no_llm_encoder):
    return ModelConfig.from_dict(ecd_config_dict_no_llm_encoder)


@pytest.fixture()
def ecd_config_dict_no_text_features():
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [{TYPE: BINARY, NAME: "in1"}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture()
def ecd_config_object_no_text_features(ecd_config_dict_no_text_features):
    return ModelConfig.from_dict(ecd_config_dict_no_text_features)


@pytest.fixture()
def gbm_config_dict_llm_encoder():
    return {
        MODEL_TYPE: MODEL_GBM,
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1", ENCODER: {TYPE: "tf_idf"}}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture()
def gbm_config_object_llm_encoder(gbm_config_dict_llm_encoder):
    return ModelConfig.from_dict(gbm_config_dict_llm_encoder)


@pytest.mark.parametrize(
    "config,expectation",
    [
        # LLM configurations
        ("llm_config_dict", True),
        ("llm_config_object", True),
        # LLM encoder configurations
        ("ecd_config_dict_llm_encoder", True),
        ("ecd_config_object_llm_encoder", True),
        # LLM encoder configurations, multiple features
        ("ecd_config_dict_llm_encoder_multiple_features", True),
        ("ecd_config_object_llm_encoder_multiple_features", True),
        # ECD configuration with text feature and non-LLM encoder
        ("ecd_config_dict_no_llm_encoder", False),
        ("ecd_config_object_no_llm_encoder", False),
        # ECD configuration with no text features
        ("ecd_config_dict_no_text_features", False),
        ("ecd_config_object_no_text_features", False),
        # GBM configuration. "tf_idf" is the only valid text encoder
        ("gbm_config_dict_llm_encoder", False),
        ("gbm_config_object_llm_encoder", False),
    ],
)
def test_is_or_uses_llm(config, expectation, request):
    config = request.getfixturevalue(config)
    assert is_or_uses_llm(config) == expectation
