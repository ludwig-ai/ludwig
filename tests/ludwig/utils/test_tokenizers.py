import pytest

from ludwig.utils.tokenizers import EnglishLemmatizeFilterTokenizer, NgramTokenizer, StringSplitTokenizer


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


def test_string_split_tokenizer():
    inputs = "Multiple,Elements,Are here!"
    tokenizer = StringSplitTokenizer(",")
    tokens = tokenizer(inputs)
    assert tokens == ["Multiple", "Elements", "Are here!"]


def test_english_lemmatize_filter_tokenizer():
    inputs = "Hello, I'm a single sentence!"
    tokenizer = EnglishLemmatizeFilterTokenizer()
    tokens = tokenizer(inputs)
    assert len(tokens) > 0


@pytest.mark.parametrize(
    "model_name,expected_cls",
    [
        # Standard BERT models must use BERTTokenizer (WordPiece)
        ("bert-base-uncased", "BERTTokenizer"),
        ("bert-large-cased", "BERTTokenizer"),
        # Models with "bert" in their name that use different tokenization
        # must NOT use BERTTokenizer
        ("roberta-base", "HFTokenizer"),
        ("albert-base-v2", "HFTokenizer"),
        ("distilbert-base-uncased", "HFTokenizer"),
        # ModernBERT uses BPE (no [UNK] token) — must NOT use BERTTokenizer
        ("answerdotai/ModernBERT-base", "HFTokenizer"),
        ("answerdotai/ModernBERT-large", "HFTokenizer"),
    ],
)
def test_get_hf_tokenizer_routing(model_name, expected_cls):
    """Regression: get_hf_tokenizer() must route ModernBERT and RoBERTa-family
    models to HFTokenizer, not BERTTokenizer.

    ModernBERT uses BPE (no [UNK] token), so loading it via BertTokenizer raises
    'WordPiece error: Missing [UNK] token from the vocabulary'.
    """
    from unittest.mock import MagicMock, patch

    from ludwig.utils.tokenizers import get_hf_tokenizer

    mock_tokenizer = MagicMock()
    with (
        patch("ludwig.utils.tokenizers.BERTTokenizer") as mock_bert,
        patch("ludwig.utils.tokenizers.HFTokenizer") as mock_hf,
    ):
        mock_bert.return_value = mock_tokenizer
        mock_hf.return_value = mock_tokenizer
        get_hf_tokenizer(model_name)

    if expected_cls == "BERTTokenizer":
        mock_bert.assert_called_once()
        mock_hf.assert_not_called()
    else:
        mock_hf.assert_called_once()
        mock_bert.assert_not_called()
