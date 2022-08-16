import pytest
import torch
import torchtext

TORCHTEXT_0_12_0_HF_NAMES = [
    # GPT2 tokenizers
    # "gpt2",  # Does not add special tokens...
    # "roberta-base",
    # "microsoft/deberta-base",
    # CLIP tokenizers
    # "microsoft/deberta-v2-xlarge"
    # SentencePiece tokenizers
    # "t5-base",  # Seems to have special tokens
    # "google/rembert",  # Seems to have special tokens
    "cointegrated/rut5-base-multitask"
]

TORCHTEXT_0_13_0_HF_NAMES = [
    "bert-base-uncased",
]


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 12, 0), reason="requires torchtext 0.12.0 or higher"
)
@pytest.mark.parametrize("pretrained_model_name_or_path", TORCHTEXT_0_12_0_HF_NAMES)
def test_hf_tokenizer_parity_torchtext_0_12_0(pretrained_model_name_or_path):
    """Tests the BERTTokenizer implementation.

    Asserts both tokens and token IDs are the same by initializing the BERTTokenizer as a standalone tokenizer and
    as a HF tokenizer.

    TODO(geoffrey): add a mixin class to GPT2Tokenizer that implements _add_special_tokens to ensure that special tokens are added the way
        that HF expects them.
    TODO(geoffrey): add a way to turn sentencepiece string tokens into integer IDs
    """
    from ludwig.utils.tokenizers import get_hf_tokenizer, HFTokenizer

    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    token_ids_expected = hf_tokenizer(inputs)

    torchtext_tokenizer = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = torchtext_tokenizer(inputs)
    print("token_ids_expected", token_ids_expected)
    print("token_ids", token_ids)
    print("tokens_expected", hf_tokenizer.tokenizer.tokenize(inputs))
    print("hf_tokenizer decode(encode())", hf_tokenizer.tokenizer.decode(hf_tokenizer.tokenizer.encode(inputs)))

    assert not isinstance(torchtext_tokenizer, HFTokenizer)
    assert token_ids_expected == token_ids


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 13, 0), reason="requires torchtext 0.13.0 or higher"
)
@pytest.mark.parametrize("pretrained_model_name_or_path", TORCHTEXT_0_13_0_HF_NAMES)
def test_hf_tokenizer_parity_torchtext_0_13_0(pretrained_model_name_or_path):
    """Tests the BERTTokenizer implementation.

    Asserts both tokens and token IDs are the same by initializing the BERTTokenizer as a standalone tokenizer and
    as a HF tokenizer.
    """
    from ludwig.utils.tokenizers import get_hf_tokenizer, HFTokenizer

    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    token_ids_expected = hf_tokenizer(inputs)

    torchtext_tokenizer = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = torchtext_tokenizer(inputs)

    assert not isinstance(torchtext_tokenizer, HFTokenizer)
    assert token_ids_expected == token_ids
