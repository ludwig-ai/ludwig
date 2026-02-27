import torch

from ludwig.accounting.used_tokens import get_used_tokens_for_ecd, get_used_tokens_for_llm


def test_get_used_tokens_for_ecd():
    inputs = {"input1": torch.tensor([[1, 2], [3, 4]]), "input2": torch.tensor([5, 6])}
    targets = {"output": torch.tensor([7, 8, 9])}

    assert get_used_tokens_for_ecd(inputs, targets) == 9


def test_get_used_tokens_for_ecd_no_targets():
    inputs = {"input1": torch.tensor([[1, 2], [3, 4]]), "input2": torch.tensor([5, 6])}
    targets = None

    assert get_used_tokens_for_ecd(inputs, targets) == 6


def test_get_used_tokens_for_llm():
    class MockTokenizer:
        pad_token_id = 0

    tokenizer = MockTokenizer()
    model_inputs = torch.tensor([1, 2, 3, 0, 0])

    assert get_used_tokens_for_llm(model_inputs, tokenizer) == 3
