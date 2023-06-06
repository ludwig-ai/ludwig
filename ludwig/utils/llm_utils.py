import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, LlamaTokenizerFast, PreTrainedTokenizer


def set_pad_token(tokenizer: PreTrainedTokenizer):
    """Sets the pad token for the tokenizer if it is not already set."""
    # Tokenizers might have the pad token id attribute since they tend to use the same base class, but
    # it can be set to None so we check for this explicitly.
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return

    # HACK(Arnav): gpt, gpt2 and llama tokenizers had no pad tokens.
    # These recommend using eos tokens instead
    # https://github.com/huggingface/transformers/issues/2648#issuecomment-616177044
    # https://github.com/huggingface/transformers/issues/2630#issuecomment-1290809338
    if any(isinstance(tokenizer, t) for t in [GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


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
