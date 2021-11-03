import pytest

import torch

from ludwig.modules.attention_modules import FeedForwardAttentionReducer


@pytest.mark.parametrize('input_hidden_size', [128, 256, 512])
@pytest.mark.parametrize('input_seq_size', [10, 20])
@pytest.mark.parametrize('input_batch_size', [16, 32])
def test_feed_forward_attention_reducer(
        input_batch_size,
        input_seq_size,
        input_hidden_size
):
    # Generate synthetic data
    current_inputs = torch.normal(
        0, 1,
        size=[input_batch_size, input_seq_size, input_hidden_size],
        dtype=torch.float32
    )

    # instantiate feed forward attention reducer
    feed_forward_attention_reducer = FeedForwardAttentionReducer(input_hidden_size)

    result = feed_forward_attention_reducer(current_inputs)

    # ensure returned tensor is the correct shape
    assert list(result.shape) == [input_batch_size, input_hidden_size]
