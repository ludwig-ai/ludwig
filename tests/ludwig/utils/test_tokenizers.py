import os

import pytest
import torch
import torchtext
from transformers.models.bert.tokenization_bert import PRETRAINED_INIT_CONFIGURATION, PRETRAINED_VOCAB_FILES_MAP

from ludwig.utils.tokenizers import NgramTokenizer, SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES


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
        for model_name in PRETRAINED_VOCAB_FILES_MAP["vocab_file"].keys()
    ],
)
def test_bert_hf_tokenizer_parity(pretrained_model_name_or_path):
    from ludwig.utils.tokenizers import BERTTokenizer, get_hf_tokenizer, HFTokenizer

    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    tokens_expected = hf_tokenizer.tokenizer.tokenize(inputs)
    token_ids_expected = hf_tokenizer(inputs)

    vocab_file = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][pretrained_model_name_or_path]
    init_kwargs = PRETRAINED_INIT_CONFIGURATION[pretrained_model_name_or_path]
    tokenizer = BERTTokenizer(vocab_file, **init_kwargs)
    tokens = tokenizer(inputs)

    tokenizer_ids_only = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = tokenizer_ids_only(inputs)

    assert not isinstance(tokenizer_ids_only, HFTokenizer)
    assert tokens == tokens_expected
    assert token_ids == token_ids_expected


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        pytest.param(
            model_name,
            marks=[
                pytest.mark.skipif(
                    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 14, 0),
                    reason="requires torchtext 0.14.0 or higher",
                ),
                pytest.mark.skipif(model_name in SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES, reason="issue on torchtext side"),
            ],
        )
        for model_name in [
            "distilbert-base-uncased",
            "google/electra-small-discriminator",
            "dbmdz/bert-base-italian-cased",
        ]
    ],
)
def test_custom_bert_hf_tokenizer_parity(tmpdir, pretrained_model_name_or_path):
    """Tests the BERTTokenizer implementation.
    Asserts both tokens and token IDs are the same by initializing the BERTTokenizer as a standalone tokenizer and as a
    HF tokenizer.
    """
    from ludwig.utils.tokenizers import get_hf_tokenizer, HFTokenizer

    inputs = "Hello, ``I'm'' ónë of 1,205,000 sentences!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    torchtext_tokenizer = get_hf_tokenizer(pretrained_model_name_or_path)

    # Ensure that the tokenizer is scriptable
    tokenizer_path = os.path.join(tmpdir, "tokenizer.pt")
    torch.jit.script(torchtext_tokenizer).save(tokenizer_path)
    torchtext_tokenizer = torch.jit.load(tokenizer_path)

    token_ids_expected = hf_tokenizer(inputs)
    token_ids = torchtext_tokenizer(inputs)

    assert token_ids_expected == token_ids


def test_ngram_tokenizer():
    inputs = "Hello, I'm a single sentence!"
    tokenizer = NgramTokenizer(n=2)
    tokens_expected = [
        "Hello,",
        "I'm",
        "a",
        "single",
        "sentence!",
        "Hello, I'm",
        "I'm a",
        "a single",
        "single sentence!",
    ]
    tokens = tokenizer(inputs)
    assert tokens == tokens_expected
