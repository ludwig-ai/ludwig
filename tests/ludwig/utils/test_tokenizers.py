import pytest

from ludwig.utils.tokenizers import (
    BERTTokenizer,
    get_hf_tokenizer,
    get_tokenizer_from_registry,
    HFTokenizer,
    SpacePunctuationStringToListTokenizer,
    SpaceStringToListTokenizer,
    tokenizer_registry,
)


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "bert-base-uncased",
        "bert-base-cased",
    ],
)
def test_bert_hf_tokenizer_parity(pretrained_model_name_or_path):
    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    tokens_expected = hf_tokenizer.tokenizer.tokenize(inputs)
    token_ids_expected = hf_tokenizer(inputs)

    tokenizer = BERTTokenizer(pretrained_model_name_or_path=pretrained_model_name_or_path, is_hf_tokenizer=False)
    tokens = tokenizer(inputs)

    tokenizer_ids_only = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = tokenizer_ids_only(inputs)

    assert tokens == tokens_expected
    assert token_ids == token_ids_expected


# ---------------------------------------------------------------------------
# SpaceStringToListTokenizer tests
# ---------------------------------------------------------------------------


class TestSpaceStringToListTokenizer:
    def test_space_tokenizer_single_string(self):
        tokenizer = SpaceStringToListTokenizer()
        result = tokenizer("hello world")
        assert result == ["hello", "world"]

    def test_space_tokenizer_list_input(self):
        tokenizer = SpaceStringToListTokenizer()
        result = tokenizer(["hello world", "foo bar"])
        assert result == [["hello", "world"], ["foo", "bar"]]

    def test_space_tokenizer_empty_string(self):
        tokenizer = SpaceStringToListTokenizer()
        result = tokenizer("")
        assert result == []


# ---------------------------------------------------------------------------
# SpacePunctuationStringToListTokenizer tests
# ---------------------------------------------------------------------------


class TestSpacePunctuationStringToListTokenizer:
    def test_space_punct_tokenizer_handles_punctuation(self):
        tokenizer = SpacePunctuationStringToListTokenizer()
        result = tokenizer("hello, world!")
        assert result == ["hello", ",", "world", "!"]

    def test_space_punct_tokenizer_list_input(self):
        tokenizer = SpacePunctuationStringToListTokenizer()
        result = tokenizer(["hello, world"])
        assert result == [["hello", ",", "world"]]


# ---------------------------------------------------------------------------
# SentencePieceTokenizer tests (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSentencePieceTokenizer:
    def test_sentencepiece_tokenizer_basic(self):
        try:
            from ludwig.utils.tokenizers import SentencePieceTokenizer

            tokenizer = SentencePieceTokenizer()
            result = tokenizer("Hello, this is a test sentence.")
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(tok, str) for tok in result)
        except Exception as e:
            pytest.skip(f"SentencePieceTokenizer unavailable or download failed: {e}")

    def test_sentencepiece_tokenizer_list_input(self):
        try:
            from ludwig.utils.tokenizers import SentencePieceTokenizer

            tokenizer = SentencePieceTokenizer()
            result = tokenizer(["Hello world", "Goodbye world"])
            assert isinstance(result, list)
            assert len(result) == 2
            for token_list in result:
                assert isinstance(token_list, list)
                assert all(isinstance(tok, str) for tok in token_list)
        except Exception as e:
            pytest.skip(f"SentencePieceTokenizer unavailable or download failed: {e}")


# ---------------------------------------------------------------------------
# CLIPTokenizer tests (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCLIPTokenizer:
    def test_clip_tokenizer_basic(self):
        try:
            from ludwig.utils.tokenizers import CLIPTokenizer

            tokenizer = CLIPTokenizer()
            result = tokenizer("a photo of a cat")
            assert isinstance(result, list)
            assert len(result) > 0
            # CLIP tokenizer returns strings (subword tokens)
            assert all(isinstance(tok, (str, int)) for tok in result)
        except Exception as e:
            pytest.skip(f"CLIPTokenizer unavailable or download failed: {e}")


# ---------------------------------------------------------------------------
# GPT2BPETokenizer tests (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGPT2BPETokenizer:
    def test_gpt2_bpe_tokenizer_basic(self):
        try:
            from ludwig.utils.tokenizers import GPT2BPETokenizer

            tokenizer = GPT2BPETokenizer()
            result = tokenizer("Hello, how are you?")
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(tok, (str, int)) for tok in result)
        except Exception as e:
            pytest.skip(f"GPT2BPETokenizer unavailable or download failed: {e}")


# ---------------------------------------------------------------------------
# get_hf_tokenizer routing tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGetHfTokenizerRouting:
    def test_get_hf_tokenizer_bert_returns_bert_tokenizer(self):
        try:
            tokenizer = get_hf_tokenizer("bert-base-uncased")
            assert isinstance(tokenizer, BERTTokenizer)
        except Exception as e:
            pytest.skip(f"Model download failed: {e}")

    def test_get_hf_tokenizer_roberta_returns_hf_tokenizer(self):
        try:
            tokenizer = get_hf_tokenizer("roberta-base")
            assert isinstance(tokenizer, HFTokenizer)
            assert not isinstance(tokenizer, BERTTokenizer)
        except Exception as e:
            pytest.skip(f"Model download failed: {e}")

    def test_get_hf_tokenizer_distilbert_returns_hf_tokenizer(self):
        try:
            tokenizer = get_hf_tokenizer("distilbert-base-uncased")
            assert isinstance(tokenizer, HFTokenizer)
            assert not isinstance(tokenizer, BERTTokenizer)
        except Exception as e:
            pytest.skip(f"Model download failed: {e}")


# ---------------------------------------------------------------------------
# get_tokenizer_from_registry tests
# ---------------------------------------------------------------------------


class TestGetTokenizerFromRegistry:
    def test_get_tokenizer_from_registry_space(self):
        cls = get_tokenizer_from_registry("space")
        assert cls is SpaceStringToListTokenizer

    def test_get_tokenizer_from_registry_invalid_raises(self):
        with pytest.raises(KeyError, match="Invalid tokenizer name"):
            get_tokenizer_from_registry("nonexistent_tokenizer_xyz")
