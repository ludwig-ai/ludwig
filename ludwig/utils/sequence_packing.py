"""Sequence packing for efficient LLM training.

Packs multiple short sequences into a single batch entry to maximize GPU utilization.
Instead of padding every sequence to max_length (wasting compute on pad tokens),
packing concatenates multiple sequences and uses a block-diagonal attention mask
to prevent cross-sequence attention.

Two strategies:
- "greedy": Simple first-fit decreasing (FFD) bin packing
- "full": Pack all sequences without gaps (may split sequences)

Config:
    trainer:
      packing: true
      packing_max_sequences_per_pack: 8  # max sequences in one pack

Based on: Krell et al., "Efficient Sequence Packing without Cross-contamination", 2021
"""

import logging

import torch

logger = logging.getLogger(__name__)


def pack_sequences(
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    labels_list: list[torch.Tensor] | None = None,
    max_length: int = 2048,
    pad_token_id: int = 0,
    max_sequences_per_pack: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Pack multiple variable-length sequences into fixed-length packs.

    Args:
        input_ids_list: List of 1D token ID tensors (unpadded).
        attention_mask_list: List of 1D attention mask tensors.
        labels_list: Optional list of 1D label tensors for loss computation.
        max_length: Maximum pack length.
        pad_token_id: Token ID used for padding.
        max_sequences_per_pack: Maximum number of sequences in one pack.

    Returns:
        Tuple of:
        - packed_input_ids: [num_packs, max_length]
        - packed_attention_mask: [num_packs, max_length, max_length] (2D block-diagonal)
        - packed_labels: [num_packs, max_length] or None (with -100 for non-label tokens)
        - sequence_ids: [num_packs, max_length] (which sequence each token belongs to)
    """
    # Sort by length descending for greedy bin packing
    lengths = [ids.shape[0] for ids in input_ids_list]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    # Greedy first-fit decreasing bin packing
    packs = []  # list of lists of (seq_index, start_pos)
    pack_remaining = []  # remaining space in each pack

    for idx in sorted_indices:
        seq_len = lengths[idx]
        if seq_len > max_length:
            # Sequence too long, truncate it and put in its own pack
            packs.append([(idx, 0)])
            pack_remaining.append(0)
            continue

        # Find first pack with enough space
        placed = False
        for pack_idx, remaining in enumerate(pack_remaining):
            if remaining >= seq_len and len(packs[pack_idx]) < max_sequences_per_pack:
                start = max_length - remaining
                packs[pack_idx].append((idx, start))
                pack_remaining[pack_idx] -= seq_len
                placed = True
                break

        if not placed:
            packs.append([(idx, 0)])
            pack_remaining.append(max_length - seq_len)

    # Build packed tensors
    device = input_ids_list[0].device
    num_packs = len(packs)

    packed_input_ids = torch.full((num_packs, max_length), pad_token_id, dtype=torch.long, device=device)
    packed_attention_mask = torch.zeros((num_packs, max_length, max_length), dtype=torch.bool, device=device)
    sequence_ids = torch.full((num_packs, max_length), -1, dtype=torch.long, device=device)
    packed_labels = None
    if labels_list is not None:
        packed_labels = torch.full((num_packs, max_length), -100, dtype=torch.long, device=device)

    for pack_idx, pack_contents in enumerate(packs):
        for seq_in_pack, (seq_idx, start_pos) in enumerate(pack_contents):
            seq_len = min(lengths[seq_idx], max_length - start_pos)
            end_pos = start_pos + seq_len

            # Copy token IDs
            packed_input_ids[pack_idx, start_pos:end_pos] = input_ids_list[seq_idx][:seq_len]

            # Block-diagonal attention: each sequence attends only to itself
            # This is a causal mask within each sequence block
            for i in range(start_pos, end_pos):
                for j in range(start_pos, i + 1):  # causal: attend to current and previous positions
                    packed_attention_mask[pack_idx, i, j] = True

            # Sequence IDs
            sequence_ids[pack_idx, start_pos:end_pos] = seq_in_pack

            # Labels
            if packed_labels is not None and labels_list is not None:
                packed_labels[pack_idx, start_pos:end_pos] = labels_list[seq_idx][:seq_len]

    logger.debug(
        f"Packed {len(input_ids_list)} sequences into {num_packs} packs "
        f"({len(input_ids_list) / max(num_packs, 1):.1f}x packing ratio)"
    )

    return packed_input_ids, packed_attention_mask, packed_labels, sequence_ids


def create_block_diagonal_mask(
    sequence_lengths: list[int],
    max_length: int,
    causal: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a block-diagonal attention mask for packed sequences.

    Args:
        sequence_lengths: Length of each sequence in the pack.
        max_length: Total pack length.
        causal: If True, apply causal masking within each block.
        device: Target device.

    Returns:
        [max_length, max_length] boolean attention mask.
    """
    mask = torch.zeros(max_length, max_length, dtype=torch.bool, device=device)
    offset = 0
    for length in sequence_lengths:
        end = min(offset + length, max_length)
        if causal:
            for i in range(offset, end):
                mask[i, offset : i + 1] = True
        else:
            mask[offset:end, offset:end] = True
        offset = end
        if offset >= max_length:
            break
    return mask
