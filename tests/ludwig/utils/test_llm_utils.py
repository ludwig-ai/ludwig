import pytest
import torch
from transformers import AutoTokenizer

from ludwig.utils.llm_utils import (
    add_left_padding,
    create_attention_mask,
    find_last_matching_index,
    has_padding_token,
    remove_left_padding,
    set_pad_token,
)

# Pad token ID is 1 for OPT even though it uses the GPT2 tokenizer
# BOS token ID is 2
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    set_pad_token(tokenizer)

    return tokenizer


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
def test_has_padding_token_with_padding_tokens(tokenizer):
    input_sentence = "This is an example sentence."
    input_ids = tokenizer([input_sentence])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"])
    padded_input_ids = torch.nn.functional.pad(input_ids["input_ids"], (10 - len(input_ids["input_ids"]), 1), value=1)

    assert has_padding_token(padded_input_ids, tokenizer)


@pytest.mark.llm
def test_has_padding_token_without_padding_tokens(tokenizer):
    input_sentence = "This is an example sentence."
    input_ids = tokenizer([input_sentence])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"])

    assert not has_padding_token(input_ids["input_ids"], tokenizer)


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
def test_remove_left_padding(input_ids, expected, tokenizer):
    assert torch.equal(remove_left_padding(input_ids, tokenizer).squeeze(0), expected)


@pytest.mark.llm
@pytest.mark.parametrize(
    "input_ids, max_length, pad_value, expected",
    [
        (torch.tensor([1, 2, 3]), 3, 0, torch.tensor([1, 2, 3])),
        (torch.tensor([1, 2, 3]), 5, 0, torch.tensor([0, 0, 1, 2, 3])),
        (torch.tensor([4, 5, 6, 7]), 6, 2, torch.tensor([2, 2, 4, 5, 6, 7])),
        (torch.tensor([8, 9]), 3, 1, torch.tensor([1, 8, 9])),
    ],
)
def test_add_left_padding(input_ids, max_length, pad_value, expected):
    padded = add_left_padding(input_ids, max_length, pad_value).squeeze(0)

    assert torch.equal(padded, expected)


@pytest.mark.llm
def test_create_attention_mask_last_token_padding(tokenizer):
    input_ids = torch.tensor([3, 4, tokenizer.pad_token_id])
    attention_mask = create_attention_mask(input_ids, tokenizer)
    assert attention_mask[-1] == 1


@pytest.mark.llm
@pytest.mark.parametrize(
    "input_ids, expected_output",
    [
        # No padding
        (torch.tensor([3, 4, 5]), torch.tensor([1, 1, 1])),
        (torch.tensor([1, 1, 4, 6, 8]), torch.tensor([0, 0, 1, 1, 1])),
        # All padding
        (torch.tensor([1, 1, 1]), torch.tensor([0, 0, 1])),
    ],
)
def test_create_attention_mask(input_ids, expected_output, tokenizer):
    attention_mask = create_attention_mask(input_ids, tokenizer)

    assert torch.equal(attention_mask, expected_output)


@pytest.mark.llm
@pytest.mark.parametrize(
    "tensor_a, tensor_b, expected_index",
    [
        # Matching index at the end
        (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([6, 7, 8]), 5),
        # No matching index
        (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([9, 10]), -1),
        # Matching index in the middle. Fails because we're only checking the last X elements.
        (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([4, 5, 6]), -1),
    ],
)
def test_find_last_matching_index(tensor_a, tensor_b, expected_index):
    last_matching_index = find_last_matching_index(tensor_a, tensor_b)
    assert last_matching_index == expected_index
