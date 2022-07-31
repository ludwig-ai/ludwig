import pytest
import torch

from ludwig.modules.attention_modules import FeedForwardAttentionReducer, MultiHeadSelfAttention
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("input_hidden_size", [128, 256, 512])
@pytest.mark.parametrize("input_seq_size", [10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_feed_forward_attention_reducer(input_batch_size, input_seq_size, input_hidden_size):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # Generate synthetic data
    current_inputs = torch.normal(0, 1, size=[input_batch_size, input_seq_size, input_hidden_size], dtype=torch.float32)

    # instantiate feed forward attention reducer
    feed_forward_attention_reducer = FeedForwardAttentionReducer(input_hidden_size)

    result = feed_forward_attention_reducer(current_inputs)

    # ensure returned tensor is the correct shape
    assert list(result.shape) == [input_batch_size, input_hidden_size]

    # check for parameter updating if fully connected layer is present
    target = torch.randn(result.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        feed_forward_attention_reducer,
        (current_inputs,),
        target,
    )
    assert upc == tpc, f"Some parameters not updated.  These parameters not updated: {not_updated}"


@pytest.mark.parametrize("input_hidden_size", [128, 256, 512])
@pytest.mark.parametrize("input_seq_size", [10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_multihead_self_attention(input_batch_size, input_seq_size, input_hidden_size):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # Generate synthetic data
    current_inputs = torch.normal(0, 1, size=[input_batch_size, input_seq_size, input_hidden_size], dtype=torch.float32)

    # instantiate feed forward attention reducer
    multihead_self_attention = MultiHeadSelfAttention(input_hidden_size, input_hidden_size)

    result = multihead_self_attention(current_inputs)

    # ensure returned tensor is the correct shape
    assert list(result.shape) == [input_batch_size, input_seq_size, input_hidden_size]

    # check for parameter updating if fully connected layer is present
    target = torch.randn(result.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        multihead_self_attention,
        (current_inputs,),
        target,
    )
    assert upc == tpc, f"Some parameters not updated.  These parameters not updated: {not_updated}"
