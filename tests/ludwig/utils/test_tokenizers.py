import os

import pytest
import torch
import torchtext

from ludwig.utils.tokenizers import EnglishLemmatizeFilterTokenizer, NgramTokenizer

TORCHTEXT_0_14_0_HF_NAMES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "dbmdz/bert-base-italian-cased",  # Community model
    "nreimers/MiniLM-L6-H384-uncased",  # Community model
    "emilyalsentzer/Bio_ClinicalBERT",  # Community model
    "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",  # Community model
]


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
            ],
        )
        for model_name in TORCHTEXT_0_14_0_HF_NAMES
    ],
)
def test_bert_hf_tokenizer_parity(tmpdir, pretrained_model_name_or_path):
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


def test_english_lemmatize_filter_tokenizer():
    inputs = "Hello, I'm a single sentence!"
    tokenizer = EnglishLemmatizeFilterTokenizer()
    tokens = tokenizer(inputs)
    assert len(tokens) > 0
