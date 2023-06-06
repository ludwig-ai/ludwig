import pytest
import torch
from transformers import AutoTokenizer

from ludwig.utils.llm_utils import remove_left_padding, set_pad_token

# Pad token ID is 1 for OPT even though it uses the GPT2 tokenizer
# BOS token ID is 2
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"


@pytest.mark.llm
def test_set_pad_token_doesnt_exist():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    assert tokenizer.pad_token_id is None

    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 50256


@pytest.mark.llm
def test_set_pad_token_already_exists():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, use_fast=False)
    assert tokenizer.pad_token_id == 1

    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 1


@pytest.mark.llm
@pytest.mark.parametrize(
    "input_ids, expected",
    [
        # No padding
        (torch.tensor([5]), torch.tensor([5])),
        (torch.tensor([5, 3]), torch.tensor([5, 3])),
        # Padding
        (torch.tensor([1, 5, 5, 3]), torch.tensor([5, 5, 3])),
        # EOS token
        (torch.tensor([2, 5, 5, 3]), torch.tensor([2, 5, 5, 3])),
        # Padding + EOS token
        (torch.tensor([1, 2, 5, 5, 3]), torch.tensor([2, 5, 5, 3])),
    ],
)
def test_remove_left_padding(input_ids, expected):
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, use_fast=False)
    set_pad_token(tokenizer)

    assert torch.equal(remove_left_padding(input_ids, tokenizer).squeeze(0), expected)
