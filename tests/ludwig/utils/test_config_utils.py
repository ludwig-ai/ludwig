from typing import Any, Dict, Optional

import pytest

from ludwig.constants import MODEL_ECD
from ludwig.schema.encoders.text_encoders import BERTConfig
from ludwig.schema.features.preprocessing.text import TextPreprocessingConfig
from ludwig.schema.model_types.utils import merge_fixed_preprocessing_params


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
    "encoder,expected",
    [
        ({"type": "parallel_cnn"}, False),
        ({"type": "bert", "trainable": False}, None),
        ({"type": "bert", "trainable": True}, False),
    ],
    ids=["parallel_cnn", "bert_fixed", "bert_trainable"],
)
def test_merge_fixed_preprocessing_params_cache_embeddings(encoder: Dict[str, Any], expected: Optional[bool]):
    preprocessing = {
        "tokenizer": "space",
        "lowercase": True,
    }

    merged_params = merge_fixed_preprocessing_params(MODEL_ECD, "text", preprocessing, encoder)
    assert merged_params.get("cache_encoder_embeddings") == expected
