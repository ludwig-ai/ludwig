"""Tests for sequence packing utilities."""

import torch

from ludwig.utils.sequence_packing import create_block_diagonal_mask, pack_sequences


class TestPackSequences:
    def test_basic_packing(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        masks = [torch.ones(3), torch.ones(2), torch.ones(1)]
        labels = [torch.tensor([10, 20, 30]), torch.tensor([40, 50]), torch.tensor([60])]

        packed_ids, packed_mask, packed_labels, seq_ids = pack_sequences(
            seqs, masks, labels, max_length=8, pad_token_id=0
        )

        # All 3 short sequences should fit in one pack (3+2+1=6 <= 8)
        assert packed_ids.shape[0] == 1
        assert packed_ids.shape[1] == 8

    def test_attention_is_block_diagonal(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        masks = [torch.ones(3), torch.ones(2)]

        packed_ids, packed_mask, _, seq_ids = pack_sequences(seqs, masks, max_length=8, pad_token_id=0)

        # Attention mask should be 2D (block diagonal)
        assert packed_mask.dim() == 3  # [num_packs, max_length, max_length]

        # Tokens from sequence 1 should NOT attend to tokens from sequence 2
        # seq1 is at positions 0-2, seq2 is at positions 3-4
        # Position 3 (first token of seq2) should NOT attend to position 2 (last token of seq1)
        assert not packed_mask[0, 3, 2].item()

        # But position 1 SHOULD attend to position 0 (same sequence, causal)
        assert packed_mask[0, 1, 0].item()

    def test_separate_packs_when_too_long(self):
        seqs = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8, 9, 10])]
        masks = [torch.ones(5), torch.ones(5)]

        packed_ids, _, _, _ = pack_sequences(seqs, masks, max_length=6, pad_token_id=0)

        # Each sequence is 5 tokens, max_length=6, so they can't fit together
        assert packed_ids.shape[0] == 2

    def test_labels_masked_correctly(self):
        seqs = [torch.tensor([1, 2, 3])]
        masks = [torch.ones(3)]
        labels = [torch.tensor([10, 20, 30])]

        _, _, packed_labels, _ = pack_sequences(seqs, masks, labels, max_length=6, pad_token_id=0)

        # Labels for actual tokens should be set, padding should be -100
        assert packed_labels[0, 0] == 10
        assert packed_labels[0, 1] == 20
        assert packed_labels[0, 2] == 30
        assert packed_labels[0, 3] == -100  # padding

    def test_sequence_ids(self):
        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        masks = [torch.ones(2), torch.ones(3)]

        _, _, _, seq_ids = pack_sequences(seqs, masks, max_length=8, pad_token_id=0)

        # Should have different sequence IDs for each packed sequence
        # and -1 for padding
        assert (seq_ids[0, :5] >= 0).all()  # 2+3 = 5 tokens have valid seq IDs
        assert (seq_ids[0, 5:] == -1).all()  # rest is padding

    def test_max_sequences_per_pack(self):
        seqs = [torch.tensor([i]) for i in range(10)]
        masks = [torch.ones(1) for _ in range(10)]

        _, _, _, _ = pack_sequences(seqs, masks, max_length=100, pad_token_id=0, max_sequences_per_pack=3)
        # 10 sequences, max 3 per pack = at least 4 packs (ceil(10/3))


class TestBlockDiagonalMask:
    def test_basic(self):
        mask = create_block_diagonal_mask([3, 2], max_length=5, causal=True)
        assert mask.shape == (5, 5)

        # First block: 3x3 lower triangular
        assert mask[0, 0].item()
        assert mask[1, 0].item()
        assert mask[1, 1].item()
        assert not mask[0, 1].item()  # causal: can't look ahead

        # Second block: positions 3-4
        assert mask[3, 3].item()
        assert mask[4, 3].item()
        assert mask[4, 4].item()

        # Cross-block: should be zero
        assert not mask[3, 2].item()
        assert not mask[4, 0].item()

    def test_non_causal(self):
        mask = create_block_diagonal_mask([2, 2], max_length=4, causal=False)
        # Non-causal: full attention within blocks
        assert mask[0, 1].item()  # position 0 can see position 1
        assert mask[1, 0].item()
        # Cross-block still zero
        assert not mask[2, 1].item()

    def test_exceeds_max_length(self):
        mask = create_block_diagonal_mask([3, 3], max_length=4)
        # Second sequence gets truncated at max_length
        assert mask.shape == (4, 4)
