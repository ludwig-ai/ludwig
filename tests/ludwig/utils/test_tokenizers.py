import pytest
import torch
import torchtext

from ludwig.utils.tokenizers import SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES, _get_bert_kwargs

BERT_TOKENIZERS = [
    "bert-base-uncased",
]

CLIP_TOKENIZERS = []

GPT2_TOKENIZERS = []

SENTENCEPIECE_TOKENIZERS = []


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        pytest.param(
            model_name,
            marks=[
                pytest.mark.skipif(
                    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 13, 0),
                    reason="requires torchtext 0.13.0 or higher",
                ),
                pytest.mark.skipif(model_name in SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES, reason="issue on torchtext side"),
            ],
        )
        for model_name in BERT_TOKENIZERS
    ],
)
def test_bert_hf_tokenizer_parity(pretrained_model_name_or_path):
    """Tests the BERTTokenizer implementation.

    Asserts both tokens and token IDs are the same by initializing the BERTTokenizer as a standalone tokenizer and
    as a HF tokenizer.
    """
    from ludwig.utils.tokenizers import BERTTokenizer, get_hf_tokenizer, HFTokenizer

    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    tokens_expected = hf_tokenizer.tokenizer.tokenize(inputs)
    token_ids_expected = hf_tokenizer(inputs)

    tokenizer_kwargs = _get_bert_kwargs(pretrained_model_name_or_path)
    tokenizer = BERTTokenizer(**tokenizer_kwargs)
    tokens = tokenizer(inputs)

    tokenizer_ids_only = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = tokenizer_ids_only(inputs)

    assert not isinstance(tokenizer_ids_only, HFTokenizer)
    assert tokens == tokens_expected
    assert token_ids == token_ids_expected
