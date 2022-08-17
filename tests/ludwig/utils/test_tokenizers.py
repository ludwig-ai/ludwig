import pytest
import torch
import torchtext


TORCHTEXT_0_13_0_HF_NAMES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
]


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 13, 0), reason="requires torchtext 0.13.0 or higher"
)
@pytest.mark.parametrize("pretrained_model_name_or_path", TORCHTEXT_0_13_0_HF_NAMES)
def test_hf_tokenizer_parity_torchtext_0_13_0(tmpdir, pretrained_model_name_or_path):
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
