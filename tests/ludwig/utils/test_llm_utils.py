import pytest
import torch
from transformers import AutoTokenizer

from ludwig.constants import LOGITS, PREDICTIONS, PROBABILITIES
from ludwig.utils.llm_utils import (
    add_left_padding,
    create_attention_mask,
    find_last_matching_index,
    generate_merged_ids,
    has_padding_token,
    pad_target_tensor_for_fine_tuning,
    realign_target_and_prediction_tensors_for_inference,
    remove_left_padding,
    set_pad_token,
)

pytestmark = [pytest.mark.llm]

# Pad token ID is 1 for OPT even though it uses the GPT2 tokenizer
# BOS token ID is 2
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    set_pad_token(tokenizer)
    return tokenizer


@pytest.fixture
def input_ids():
    # Provide sample input IDs tensor
    return torch.tensor([[3, 4, 5], [6, 7, 8]])


@pytest.fixture
def target_ids():
    # Provide sample target IDs tensor
    return torch.tensor([[9, 10, 11], [12, 13, 14]])


def test_set_pad_token_doesnt_exist():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    assert tokenizer.pad_token_id is None

    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 50256


def test_set_pad_token_already_exists():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, use_fast=False)
    assert tokenizer.pad_token_id == 1

    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 1


def test_has_padding_token_with_padding_tokens(tokenizer):
    input_sentence = "This is an example sentence."
    input_ids = tokenizer([input_sentence])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"])
    padded_input_ids = torch.nn.functional.pad(input_ids["input_ids"], (10 - len(input_ids["input_ids"]), 1), value=1)

    assert has_padding_token(padded_input_ids, tokenizer)


def test_has_padding_token_without_padding_tokens(tokenizer):
    input_sentence = "This is an example sentence."
    input_ids = tokenizer([input_sentence])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"])

    assert not has_padding_token(input_ids["input_ids"], tokenizer)


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


def test_create_attention_mask_last_token_padding(tokenizer):
    input_ids = torch.tensor([3, 4, tokenizer.pad_token_id])
    attention_mask = create_attention_mask(input_ids, tokenizer)
    assert attention_mask[-1] == 1


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


def test_generate_merged_ids_with_target(tokenizer, input_ids, target_ids):
    # Test case when target_ids is not None
    merged_ids, attention_masks = generate_merged_ids(input_ids, target_ids, tokenizer)
    assert torch.equal(merged_ids, torch.tensor([[3, 4, 5, 9, 10, 11, 1], [6, 7, 8, 12, 13, 14, 1]]))
    assert merged_ids.shape == (2, 7)  # Check the shape of merged_ids
    assert attention_masks.shape == (2, 7)  # Check the shape of attention_masks


def test_generate_merged_ids_with_max_sequence_length(tokenizer, input_ids, target_ids):
    # Test case when max_sequence_length is provided
    max_sequence_length = 5
    merged_ids, attention_masks = generate_merged_ids(input_ids, target_ids, tokenizer, max_sequence_length)
    assert merged_ids.shape == (2, 5)  # Check the shape of merged_ids with truncation
    assert attention_masks.shape == (2, 5)  # Check the shape of attention_masks


def test_generate_merged_ids_padding_removal(tokenizer, input_ids, target_ids):
    # Test case to check removal of left padding from inputs and targets during merge
    padding_tokens = torch.tensor([tokenizer.pad_token_id, tokenizer.pad_token_id])

    # Adds 2 padding tokens to the left of input_ids and target_ids individually. Typically, if we just merged this
    # naively, we would expect to see [1, 1, 3, 4, 5, 1, 1, 9, 10, 11, 1], but we shouldn't see the padding tokens
    # except for the padding token at the end.
    input_ids_with_padding = torch.cat((padding_tokens.unsqueeze(0).expand(input_ids.size(0), -1), input_ids), dim=1)
    target_ids_with_padding = torch.cat((padding_tokens.unsqueeze(0).expand(target_ids.size(0), -1), target_ids), dim=1)

    merged_ids, attention_masks = generate_merged_ids(input_ids_with_padding, target_ids_with_padding, tokenizer)

    assert merged_ids.shape == (2, 7)  # Check the shape of merged_ids
    assert attention_masks.shape == (2, 7)  # Check the shape of attention_masks

    assert torch.equal(merged_ids[0][:3], input_ids[0])  # Check the input_ids part without padding
    assert torch.equal(merged_ids[0][3:-1], target_ids[0])  # Check the target_ids part without padding
    assert torch.equal(merged_ids[0][-1], torch.tensor(tokenizer.pad_token_id))  # Check the padding tokens

    assert torch.all(attention_masks == 1)


def test_generate_merged_ids_returns_tensor(tokenizer, input_ids, target_ids):
    # Test that the function returns torch.Tensor objects
    merged_ids, attention_masks = generate_merged_ids(input_ids, target_ids, tokenizer)
    assert isinstance(merged_ids, torch.Tensor)
    assert isinstance(attention_masks, torch.Tensor)


def test_pad_target_tensor_for_fine_tuning():
    of_name = "out_1"
    prediction = {
        of_name: {PREDICTIONS: torch.tensor([[764, 764, 764, 764, 764, 764, 764, 578, 619, 841, 182, 905, 483, 764]])}
    }

    # Scenario 1: Entire target tensor was passed into model inputs
    model_input = torch.tensor([[0, 0, 24, 52, 654, 529, 221, 78, 79, 504, 76, 397, 84, 0]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, 78, 79, 504, 76, 397, 84, 0]])}
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])

    # Scenario 2: Entire target tensor was not passed into model inputs
    model_input = torch.tensor([[13, 24, 395, 13, 46, 57, 52, 41, 45, 37, 51, 14, 380, 435]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {
        of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
    }
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])

    # Scenario 3: Partial target tensor was passed into model inputs
    model_input = torch.tensor([[0, 0, 24, 52, 654, 529, 221, 78, 79, 504, 76, 78, 79, 504]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {
        of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 78, 79, 504]])
    }
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])


def test_realign_target_and_prediction_tensors_for_inference(tokenizer):
    of_name = "out_1"
    vocab_size = 8

    # Scenario 1: Prediction and target tensors have the same length, so nothing should change
    targets = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    predictions = {
        of_name: {
            PREDICTIONS: torch.tensor([[78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64),
            PROBABILITIES: torch.randn(1, 7, vocab_size).to(torch.float32),
            LOGITS: torch.randn(1, 7, vocab_size).to(torch.float32),
        }
    }
    updated_targets, updated_predictions = realign_target_and_prediction_tensors_for_inference(
        targets, predictions, of_name, tokenizer
    )

    assert targets == updated_targets
    assert predictions == updated_predictions
    assert predictions[of_name][PREDICTIONS].shape[1] == targets[of_name].shape[1]
    assert predictions[of_name][PROBABILITIES].shape[1] == targets[of_name].shape[1]
    assert predictions[of_name][LOGITS].shape[1] == targets[of_name].shape[1]

    # Scenario 2: Prediction length is longer than the target tensor, so we need to realign the target tensor
    targets = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    predictions = {
        of_name: {
            PREDICTIONS: torch.tensor([[98, 47, 78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64),
            PROBABILITIES: torch.randn(1, 9, vocab_size).to(torch.float32),
            LOGITS: torch.randn(1, 9, vocab_size).to(torch.float32),
        }
    }
    updated_targets, updated_predictions = realign_target_and_prediction_tensors_for_inference(
        targets, predictions, of_name, tokenizer
    )

    assert predictions == updated_predictions
    assert torch.equal(updated_targets[of_name], torch.tensor([[78, 79, 504, 76, 397, 84, 0, 1, 1]]))

    # Scenario 3: Target length is longer than the prediction tensor, so we need to realign them
    targets = {of_name: torch.tensor([[98, 47, 78, 79, 504, 76, 397, 84, 0]])}
    predictions = {
        of_name: {
            PREDICTIONS: torch.tensor([[78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64),
            PROBABILITIES: torch.randn(1, 7, vocab_size).to(torch.float32),
            LOGITS: torch.randn(1, 7, vocab_size).to(torch.float32),
        }
    }
    updated_targets, updated_predictions = realign_target_and_prediction_tensors_for_inference(
        targets, predictions, of_name, tokenizer
    )

    assert targets == updated_targets

    assert torch.equal(updated_predictions[of_name][PREDICTIONS], torch.tensor([[78, 79, 504, 76, 397, 84, 0, 1, 1]]))
    assert updated_predictions[of_name][PROBABILITIES].shape[1] == targets[of_name].shape[1]
    assert updated_predictions[of_name][LOGITS].shape[1] == targets[of_name].shape[1]

    assert torch.equal(updated_predictions[of_name][PROBABILITIES][0][-1], torch.zeros(vocab_size))
    assert torch.equal(updated_predictions[of_name][PROBABILITIES][0][-2], torch.zeros(vocab_size))
    assert not torch.equal(updated_predictions[of_name][PROBABILITIES][0][-3], torch.zeros(vocab_size))

    assert torch.equal(updated_predictions[of_name][LOGITS][0][-1], torch.zeros(vocab_size))
    assert torch.equal(updated_predictions[of_name][LOGITS][0][-2], torch.zeros(vocab_size))
    assert not torch.equal(updated_predictions[of_name][LOGITS][0][-3], torch.zeros(vocab_size))
