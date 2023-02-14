# from typing import Any, Dict, Optional

# import pytest

# from ludwig.utils.config_utils import merge_fixed_preprocessing_params


# @pytest.mark.parametrize(
#     "pretrained_model_name_or_path",
#     [None, "bert-large-uncased"],
#     ids=["default_model", "override_model"],
# )
# def test_merge_fixed_preprocessing_params(pretrained_model_name_or_path: str):
#     expected_model_name = "bert-base-uncased"

#     preprocessing = {
#         "tokenizer": "space",
#         "lowercase": True,
#     }

#     encoder = {"type": "bert"}
#     if pretrained_model_name_or_path is not None:
#         encoder["pretrained_model_name_or_path"] = pretrained_model_name_or_path
#         expected_model_name = pretrained_model_name_or_path

#     merged_params = merge_fixed_preprocessing_params("text", preprocessing, encoder)
#     assert merged_params == {
#         "tokenizer": "hf_tokenizer",
#         "lowercase": True,
#         "pretrained_model_name_or_path": expected_model_name,
#     }


# @pytest.mark.parametrize(
#     "encoder,expected",
#     [
#         ({"type": "parallel_cnn"}, False),
#         ({"type": "bert", "trainable": False}, None),
#         ({"type": "bert", "trainable": True}, False),
#     ],
#     ids=["parallel_cnn", "bert_fixed", "bert_trainable"],
# )
# def test_merge_fixed_preprocessing_params_cache_embeddings(encoder: Dict[str, Any], expected: Optional[bool]):
#     preprocessing = {
#         "tokenizer": "space",
#         "lowercase": True,
#     }

#     merged_params = merge_fixed_preprocessing_params("text", preprocessing, encoder)
#     assert merged_params.get("cache_encoder_embeddings") == expected
