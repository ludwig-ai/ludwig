from typing import Dict, Union

import torch


def get_used_tokens_for_gbm(inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> int:
    """Returns the number of used tokens for a GBM model.

    The number of used tokens is:
    1. the size of the input tensor, which corresponds to 1 token for each input feature
    (binary, category, number) in the batch.
    2. batch_size, which corresponds to 1 token for the batch of target features.

    Args:
        inputs: The input tensors that are fed to the gbm.forward() method.
    """
    if isinstance(inputs, torch.Tensor):
        # Inputs may be a tensor for evaluation.
        # Use the total number of inputs + the batch size as the number of output tokens.
        return torch.flatten(inputs).shape[0] + inputs.shape[0]
    return len(inputs.keys()) + 1


def get_used_tokens_for_ecd(inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> int:
    """Returns the number of used tokens for an ECD model.

    The number of used tokens is the total size of the input and output tensors, which corresponds to 1 token for
    binary, category, and number features, and variable number of tokens for text features, for each example in the
    batch.

    Args:
        inputs: The input tensors for one forward pass through ecd.
        targets: The target tensors for one forward pass through ecd.
    """
    used_tokens = 0
    for input_feature_tensor in inputs.values():
        used_tokens += torch.flatten(input_feature_tensor).shape[0]
    if targets is not None:
        # targets may be None for evaluation.
        for output_feature_tensor in targets.values():
            used_tokens += torch.flatten(output_feature_tensor).shape[0]
    return used_tokens


def get_used_tokens_for_llm(model_inputs: torch.Tensor, tokenizer) -> int:
    """Returns the number of used tokens for an LLM model.

    Args:
        model_inputs: torch.Tensor with the merged input and target IDs.
        tokenizer: The tokenizer used to encode the inputs.

    Returns:
        The total number of non-pad tokens, for all examples in the batch.
    """
    return torch.sum(model_inputs != tokenizer.pad_token_id).item()
