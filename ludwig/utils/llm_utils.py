import torch
from transformers import PreTrainedTokenizer


def has_padding_token(input_tensor: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Returns True if the input tensor has padding.

    TODO(Arnav): Add example of both possible return values.
    """
    return torch.any(input_tensor == tokenizer.pad_token_id).item()


def find_last_matching_index(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    """Returns the last index of tensor_a that matches tensor_b.

    TODO(Arnav): Add example of both possible return values.
    """
    last_index = -1

    tensor_a_length = tensor_a.shape[0]
    tensor_b_length = tensor_b.shape[0]

    # Get the last tensor_b_length elements of tensor_a.
    tensor_a_truncated = tensor_a[-tensor_b_length:]

    # Find the last matching index.
    for i in range(tensor_b_length):
        if torch.equal(tensor_a_truncated[i:], tensor_b[: tensor_b_length - i]):
            last_index = tensor_a_length - tensor_b_length + i
            break

    return last_index
