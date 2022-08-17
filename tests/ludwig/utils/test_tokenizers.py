import os

import pytest
import torch
import torchtext


TORCHTEXT_0_13_0_HF_NAMES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "nreimers/MiniLM-L6-H384-uncased",  # Community model
    # "dbmdz/bert-base-italian-cased",  # Community model. Skipped: https://github.com/pytorch/text/issues/1840
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
