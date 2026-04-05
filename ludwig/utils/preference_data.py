"""Preference data handling for DPO/KTO/ORPO/GRPO training.

Provides utilities for processing preference data where each example has:
- A prompt (input)
- A chosen completion (preferred response)
- A rejected completion (dispreferred response)

In Ludwig's data model, preference data is represented as:
- Input feature: the prompt text
- Output feature: the chosen completion
- Additional column: the rejected completion (specified via trainer config)

The DPO/KTO/ORPO trainers access rejected completions during training to
compute preference losses.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def tokenize_preference_pair(
    tokenizer,
    prompt_ids: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize and merge prompt with chosen and rejected completions separately.

    Returns merged sequences for both chosen and rejected, each concatenating
    the prompt with its respective completion.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt_ids: [batch, prompt_len] token IDs for the prompt
        chosen_ids: [batch, chosen_len] token IDs for the chosen completion
        rejected_ids: [batch, rejected_len] token IDs for the rejected completion
        max_length: Maximum sequence length

    Returns:
        Dict with:
        - chosen_input_ids: [batch, max_length]
        - chosen_attention_mask: [batch, max_length]
        - chosen_labels: [batch, max_length] (with -100 for prompt tokens)
        - rejected_input_ids: [batch, max_length]
        - rejected_attention_mask: [batch, max_length]
        - rejected_labels: [batch, max_length] (with -100 for prompt tokens)
    """
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def merge_and_pad(prompt, completion):
        """Merge prompt + completion, pad to max_length, create labels with prompt masked."""
        merged_ids = []
        merged_masks = []
        merged_labels = []

        for i in range(batch_size):
            # Remove padding from both
            p = prompt[i][prompt[i] != pad_token_id]
            c = completion[i][completion[i] != pad_token_id]

            # Concatenate
            combined = torch.cat([p, c])[:max_length]
            seq_len = combined.shape[0]

            # Pad to max_length (left padding)
            pad_len = max_length - seq_len
            padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, device=device), combined])
            padded_mask = torch.cat([torch.zeros(pad_len, device=device), torch.ones(seq_len, device=device)])

            # Labels: mask prompt tokens with -100, keep completion tokens
            prompt_len = min(p.shape[0], max_length)
            labels = padded_ids.clone()
            labels[:pad_len + prompt_len] = -100  # mask padding + prompt

            merged_ids.append(padded_ids)
            merged_masks.append(padded_mask)
            merged_labels.append(labels)

        return (
            torch.stack(merged_ids).long(),
            torch.stack(merged_masks).long(),
            torch.stack(merged_labels).long(),
        )

    chosen_input_ids, chosen_attention_mask, chosen_labels = merge_and_pad(prompt_ids, chosen_ids)
    rejected_input_ids, rejected_attention_mask, rejected_labels = merge_and_pad(prompt_ids, rejected_ids)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }


def validate_preference_data(df, chosen_column: str, rejected_column: str):
    """Validate that a dataframe has the required preference columns.

    Args:
        df: Pandas DataFrame
        chosen_column: Name of the column with chosen completions
        rejected_column: Name of the column with rejected completions

    Raises:
        ValueError if columns are missing or empty
    """
    if chosen_column not in df.columns:
        raise ValueError(
            f"Preference training requires a '{chosen_column}' column in the data. "
            f"Available columns: {list(df.columns)}"
        )
    if rejected_column not in df.columns:
        raise ValueError(
            f"Preference training requires a '{rejected_column}' column in the data. "
            f"Available columns: {list(df.columns)}"
        )

    null_chosen = df[chosen_column].isna().sum()
    null_rejected = df[rejected_column].isna().sum()
    if null_chosen > 0:
        logger.warning(f"{null_chosen} rows have null values in '{chosen_column}' column")
    if null_rejected > 0:
        logger.warning(f"{null_rejected} rows have null values in '{rejected_column}' column")
