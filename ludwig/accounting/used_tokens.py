from typing import Dict

import torch


def get_used_tokens_for_gbm(inputs: Dict[str, torch.Tensor]) -> int:
    """Returns the number of used tokens for a GBM model."""
    # 1 token for each input, and 1 token for the target.
    if isinstance(inputs, torch.Tensor):
        # Inputs may be a tensor for evaluation.
        # Use the total number of inputs + the batch size as the number of output tokens.
        return torch.flatten(inputs).shape[0] + inputs.shape[0]
    return len(inputs.keys()) + 1


def get_used_tokens_for_ecd(inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> int:
    """Returns the number of used tokens for an ECD model."""
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

    model_inputs: torch.Tensor with the merged input and target IDs.
    """
    return torch.sum(model_inputs != tokenizer.pad_token_id).item()
