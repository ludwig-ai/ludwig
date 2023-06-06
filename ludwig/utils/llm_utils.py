import torch
from transformers import PreTrainedTokenizer


def remove_left_padding(input_ids_sample: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Removes left padding and other tokens until the first BOS token from the input_ids tensor."""
    # Remove all PAD tokens
    pad_idxs = torch.where(input_ids_sample == tokenizer.pad_token_id)[0]  # all PAD token locations
    input_ids_no_padding = input_ids_sample
    if len(pad_idxs) != 0:
        pad_idx = pad_idxs[-1]  # get last PAD token location
        input_ids_no_padding = input_ids_sample[pad_idx + 1 :]

    # Start from the first BOS token
    bos_idxs = torch.where(input_ids_no_padding == tokenizer.bos_token_id)[0]  # all BOS token locations
    if len(bos_idxs) != 0:
        bos_idx = bos_idxs[0]  # get first BOS token location
    else:
        bos_idx = 0

    input_ids_no_bos = input_ids_no_padding[bos_idx:].unsqueeze(0)
    return input_ids_no_bos
