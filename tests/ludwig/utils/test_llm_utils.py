import pytest
import torch
from transformers import AutoTokenizer

from ludwig.constants import PREDICTIONS
from ludwig.utils.llm_utils import (  # realign_target_and_prediction_tensors_for_inference,
    add_left_padding,
    create_attention_mask,
    find_last_matching_index,
    has_padding_token,
    pad_target_tensor_for_fine_tuning,
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


@pytest.mark.llm
def test_pad_target_tensor_for_fine_tuning():
    of_name = "out_1"

    # Scenario 1: Entire target tensor was passed into model inputs
    model_input = torch.tensor(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                52,
                654,
                332,
                664,
                88,
                84,
                529,
                318,
                40,
                45,
                50,
                35,
                67,
                494,
                312,
                383,
                381,
                79,
                589,
                364,
                293,
                89,
                518,
                599,
                769,
                380,
                435,
                311,
                529,
                221,
                78,
                79,
                504,
                76,
                397,
                84,
                0,
            ]
        ]
    )
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    prediction = {
        of_name: {
            PREDICTIONS: torch.tensor(
                [
                    [
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        764,
                        600,
                        332,
                        686,
                        717,
                        869,
                        325,
                        91,
                        166,
                        153,
                        686,
                        285,
                        622,
                        869,
                        139,
                        621,
                        376,
                        622,
                        622,
                        1023,
                        725,
                        869,
                        783,
                        401,
                        300,
                        829,
                        621,
                        981,
                        808,
                        91,
                        300,
                        578,
                        619,
                        841,
                        182,
                        905,
                        483,
                        764,
                    ]
                ]
            )
        }
    }
    expected_target = {
        of_name: torch.tensor(
            [
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    78,
                    79,
                    504,
                    76,
                    397,
                    84,
                    0,
                ]
            ]
        )
    }
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])

    # Scenario 2: Entire target tensor was not passed into model inputs
    model_input = torch.tensor(
        [
            [
                52,
                654,
                332,
                664,
                88,
                84,
                529,
                318,
                43,
                82,
                396,
                65,
                45,
                280,
                541,
                48,
                635,
                563,
                921,
                470,
                298,
                337,
                470,
                481,
                825,
                391,
                1,
                329,
                70,
                470,
                298,
                734,
                509,
                290,
                747,
                67,
                311,
                12,
                470,
                298,
                747,
                861,
                310,
                494,
                312,
                262,
                310,
                594,
                382,
                308,
                13,
                24,
                395,
                13,
                46,
                57,
                52,
                41,
                45,
                37,
                51,
                14,
                380,
                435,
            ]
        ]
    )
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {
        of_name: torch.tensor(
            [
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                ]
            ]
        )
    }
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])

    # Scenario 3: Partial target tensor was passed into model inputs
    model_input = torch.tensor(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                52,
                654,
                332,
                664,
                88,
                84,
                529,
                318,
                40,
                45,
                50,
                35,
                67,
                494,
                312,
                383,
                381,
                79,
                589,
                364,
                293,
                89,
                518,
                599,
                769,
                380,
                435,
                311,
                529,
                221,
                123,
                664,
                79,
                23,
                78,
                79,
                504,
            ]
        ]
    )
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {
        of_name: torch.tensor(
            [
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    78,
                    79,
                    504,
                ]
            ]
        )
    }
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])
