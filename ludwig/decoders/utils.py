from typing import List

import torch
from torch import Tensor


def extract_generated_tokens(
    raw_generated_output_sequences: List[Tensor],
    input_lengths: List[int],
    max_new_tokens: int,
    pad_sequence: bool,
) -> List[Tensor]:
    """Extracts the generated tokens from the raw output sequences of the language model.

    Args:
        raw_generated_output_sequences: The raw output sequences of the language model.
            Represented as a list to handle variable length sequences.
        input_lengths: The length of the inputs to the language model.
        max_new_tokens: The maximum number of new tokens that were generated. Used to
            pad the generated sequences to the max_new_tokens.
        pad_sequence: Whether to pad the generated sequences to the max_new_tokens.

    Returns:
        The generated tokens.
    """
    if len(raw_generated_output_sequences) != len(input_lengths):
        raise ValueError(
            f"The number of raw_generated_output_sequences ({len(raw_generated_output_sequences)}) "
            f"must be the same as the number of input_lengths ({len(input_lengths)})."
        )

    generated_outputs = []
    for idx, input_length in enumerate(input_lengths):
        # Remove the input sequence from the generated sequence
        generated_sequence = raw_generated_output_sequences[idx][input_length:]

        # Pad the sequence if it is shorter than the max_new_tokens for downstream metric computation
        if pad_sequence and generated_sequence.size()[0] < max_new_tokens:
            generated_sequence = torch.nn.functional.pad(
                generated_sequence, (0, max_new_tokens - generated_sequence.size()[0]), "constant", 0
            )
        generated_outputs.append(generated_sequence)
    return generated_outputs
