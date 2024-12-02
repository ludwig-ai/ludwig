import copy
from typing import Any, Dict, List, Optional, Union

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
from ludwig.utils.config_utils import config_uses_llm, get_quantization


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


@pytest.fixture(scope="module")
def llm_config_dict() -> Dict[str, Any]:
    return {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1"}],
        OUTPUT_FEATURES: [{TYPE: TEXT, NAME: "out1"}],
    }


@pytest.fixture(scope="module")
def ecd_config_dict_llm_encoder() -> Dict[str, Any]:
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


@pytest.fixture(scope="module")
def ecd_config_dict_llm_encoder_multiple_features() -> Dict[str, Any]:
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


@pytest.fixture(scope="module")
def ecd_config_dict_no_llm_encoder() -> Dict[str, Any]:
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1", ENCODER: {TYPE: "parallel_cnn"}}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture(scope="module")
def ecd_config_dict_no_text_features() -> Dict[str, Any]:
    return {
        MODEL_TYPE: MODEL_ECD,
        INPUT_FEATURES: [{TYPE: BINARY, NAME: "in1"}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture(scope="module")
def gbm_config_dict() -> Dict[str, Any]:
    return {
        MODEL_TYPE: MODEL_GBM,
        INPUT_FEATURES: [{TYPE: TEXT, NAME: "in1", ENCODER: {TYPE: "tf_idf"}}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.fixture(scope="module")
def gbm_config_dict_no_text_features() -> Dict[str, Any]:
    return {
        MODEL_TYPE: MODEL_GBM,
        INPUT_FEATURES: [{TYPE: BINARY, NAME: "in1"}],
        OUTPUT_FEATURES: [{TYPE: BINARY, NAME: "out1"}],
    }


@pytest.mark.parametrize(
    "config,expectation",
    [
        # LLM configurations
        ("llm_config_dict", True),
        # LLM encoder configurations
        ("ecd_config_dict_llm_encoder", True),
        # LLM encoder configurations, multiple features
        ("ecd_config_dict_llm_encoder_multiple_features", True),
        # ECD configuration with text feature and non-LLM encoder
        ("ecd_config_dict_no_llm_encoder", False),
        # ECD configuration with no text features
        ("ecd_config_dict_no_text_features", False),
        # GBM configuration with text feature. "tf_idf" is the only valid text encoder
        ("gbm_config_dict", False),
        # GBM configuration with no text features
        ("gbm_config_dict_no_text_features", False),
    ],
)
@pytest.mark.parametrize("config_type", ["dict", "object"])
def test_is_or_uses_llm(config: Dict[str, Any], expectation: bool, config_type, request):
    """Test LLM detection on a variety of configs. Configs that use an LLM anywhere should return True, otherwise
    False.

    Args:
        config: The name of the config fixture to test
        expectation: The expected result
        request: pytest `request` fixture
    """
    config = request.getfixturevalue(config)
    if config_type == "object":
        config = ModelConfig.from_dict(config)
    assert config_uses_llm(config) == expectation


@pytest.mark.parametrize("invalid_config", [1, 1.0, "foo", True, False, None, [], {}, {"foo": "bar"}])
def test_is_or_uses_llm_invalid_input(invalid_config):
    """Sanity checks for invalid config handling.

    These should all raise an exception.

    Args:
        invalid_config: An invalid argument to `config_uses_llm`
    """
    with pytest.raises(ValueError):
        config_uses_llm(invalid_config)


@pytest.fixture(scope="module")
def quantization_4bit_config() -> Dict[str, Any]:
    return {"quantization": {"bits": 4}}


@pytest.fixture(scope="module")
def quantization_8bit_config() -> Dict[str, Any]:
    return {"quantization": {"bits": 8}}


@pytest.fixture(scope="module")
def llm_config_dict_4bit(llm_config_dict: Dict[str, Any], quantization_4bit_config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(llm_config_dict)
    config.update(quantization_4bit_config)
    return config


@pytest.fixture(scope="module")
def llm_config_dict_8bit(llm_config_dict: Dict[str, Any], quantization_8bit_config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(llm_config_dict)
    config.update(quantization_8bit_config)
    return config


@pytest.fixture(scope="module")
def ecd_config_dict_llm_encoder_4bit(
    ecd_config_dict_llm_encoder: Dict[str, Any], quantization_4bit_config: Dict[str, Any]
) -> Dict[str, Any]:
    config = copy.deepcopy(ecd_config_dict_llm_encoder)
    config[INPUT_FEATURES][0][ENCODER].update(quantization_4bit_config)
    return config


@pytest.fixture(scope="module")
def ecd_config_dict_llm_encoder_8bit(
    ecd_config_dict_llm_encoder: Dict[str, Any], quantization_8bit_config: Dict[str, Any]
) -> Dict[str, Any]:
    config = copy.deepcopy(ecd_config_dict_llm_encoder)
    config[INPUT_FEATURES][0][ENCODER].update(quantization_8bit_config)
    return config


@pytest.mark.parametrize(
    "config,expectation",
    [
        # LLM configurations
        ("llm_config_dict", [None]),
        ("llm_config_dict_4bit", [4]),
        ("llm_config_dict_8bit", [8]),
        # LLM encoder configurations with one feature
        ("ecd_config_dict_llm_encoder", [None]),
        ("ecd_config_dict_llm_encoder_4bit", [4]),
        ("ecd_config_dict_llm_encoder_8bit", [8]),
        # GBM configuration with text feature. "tf_idf" is the only valid text encoder
        ("gbm_config_dict", [None]),
        # GBM configuration with no text features
        ("gbm_config_dict_no_text_features", [None]),
    ],
)
@pytest.mark.parametrize("config_type", ["dict", "object"])
def test_get_quantization(
    config: Dict[str, Any], expectation: Union[int, List[int], None, List[None]], config_type: str, request
):
    """Test get_quantization with LLM and single-feature ECD/GBM configs.

    Args:
        config: The configuration to test
        expectation: The expected quantization
        config_type: Whether to test the config as a dict or object
        request: pytest builtin fixture
    """
    config = request.getfixturevalue(config)
    if config_type == "object":
        config = ModelConfig.from_dict(config)
    assert get_quantization(config) == expectation


TEST_FEATURE_CONFIGS = [
    (
        {
            TYPE: BINARY,
        },
        None,
    ),
    (
        {
            TYPE: TEXT,
        },
        None,
    ),
    ({TYPE: TEXT, ENCODER: {TYPE: MODEL_LLM, BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM"}}, None),
    (
        {
            TYPE: TEXT,
            ENCODER: {
                TYPE: MODEL_LLM,
                BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
                "quantization": {"bits": 4},
            },
        },
        4,
    ),
    (
        {
            TYPE: TEXT,
            ENCODER: {
                TYPE: MODEL_LLM,
                BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
                "quantization": {"bits": 8},
            },
        },
        8,
    ),
]

TEST_FEATURE_CONFIGS_IDS = [BINARY, TEXT, MODEL_LLM, f"{MODEL_LLM}-4bit", f"{MODEL_LLM}-8bit"]


@pytest.mark.parametrize("feature1,quantization1", TEST_FEATURE_CONFIGS, ids=TEST_FEATURE_CONFIGS_IDS)
@pytest.mark.parametrize("feature2,quantization2", TEST_FEATURE_CONFIGS, ids=TEST_FEATURE_CONFIGS_IDS)
@pytest.mark.parametrize("config_type", ["dict", "object"])
def test_get_quantization_multiple_features(
    ecd_config_dict_llm_encoder_multiple_features: Dict[str, Any],
    feature1: Dict[str, Any],
    quantization1: int,
    feature2: Dict[str, Any],
    quantization2: int,
    config_type: str,
):
    """Test get_quantization with multiple features.

    Args:
        ecd_config_dict_llm_encoder_multiple_features: Baseline config to add features to.
        feature1: First input feature config dict
        quantization1: First input feature expected quantization
        feature2: Second input feature config dict
        quantization2: Second input feature expected quantization
        config_type: Whether to test the config as a dict or object
    """
    config = copy.deepcopy(ecd_config_dict_llm_encoder_multiple_features)
    feature1 = dict(name="in1", **feature1)
    feature2 = dict(name="in2", **feature2)
    config[INPUT_FEATURES] = [feature1, feature2]

    if config_type == "object":
        config = ModelConfig.from_dict(config)

    assert get_quantization(config) == [quantization1, quantization2]


@pytest.mark.parametrize("invalid_config", [1, 1.0, "foo", True, False, None, [], {}, {"foo": "bar"}])
def test_get_quantization_invalid_input(invalid_config):
    """Test get_quantization with invalid configs. These should always raise a ValueError.

    Args:
        invalid_config: The invalid config to test
    """
    with pytest.raises(ValueError):
        get_quantization(invalid_config)
