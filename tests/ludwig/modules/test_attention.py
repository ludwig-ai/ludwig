import pytest
import torch

from ludwig.modules.attention_modules import (
    FeedForwardAttentionReducer,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerStack,
)
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("input_hidden_size", [128, 256, 512])
@pytest.mark.parametrize("input_seq_size", [10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_feed_forward_attention_reducer(input_batch_size: int, input_seq_size: int, input_hidden_size: int):
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
@pytest.mark.parametrize("input_seq_size", [1, 10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_multihead_self_attention(input_batch_size: int, input_seq_size: int, input_hidden_size: int):
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

    # adjustment required for single token sequence because self attention query and key parameters are
    # not updated
    single_sequence_token_adjustment = 4 if input_seq_size == 1 else 0
    assert upc == (
        tpc - single_sequence_token_adjustment
    ), f"Some parameters not updated.  These parameters not updated: {not_updated}"


# heads must be a divisor of input_hidden_size
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("output_size", [64, 128, 256])
@pytest.mark.parametrize("input_hidden_size", [128, 256, 512])
@pytest.mark.parametrize("input_seq_size", [10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_transformer_block(
    input_batch_size: int,
    input_seq_size: int,
    input_hidden_size: int,
    output_size: int,
    heads: int,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # Generate synthetic data
    current_inputs = torch.normal(0, 1, size=[input_batch_size, input_seq_size, input_hidden_size], dtype=torch.float32)

    # instantiate feed forward attention reducer
    transformer_block = TransformerBlock(input_hidden_size, input_seq_size, input_hidden_size, heads, output_size)

    result = transformer_block(current_inputs)

    # ensure returned tensor is the correct shape
    assert list(result.shape) == [input_batch_size, input_seq_size, input_hidden_size]

    # check for parameter updating if fully connected layer is present
    target = torch.randn(result.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        transformer_block,
        (current_inputs,),
        target,
    )
    assert upc == tpc, f"Some parameters not updated.  These parameters not updated: {not_updated}"


@pytest.mark.parametrize("num_layers", [1, 4])
# heads must be a divisor of input_hidden_size
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("output_size", [64, 128, 256])
@pytest.mark.parametrize("input_hidden_size", [128, 256, 512])
@pytest.mark.parametrize("input_seq_size", [10, 20])
@pytest.mark.parametrize("input_batch_size", [16, 32])
def test_transformer_stack(
    input_batch_size: int,
    input_seq_size: int,
    input_hidden_size: int,
    output_size: int,
    heads: int,
    num_layers: int,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # Generate synthetic data
    current_inputs = torch.normal(0, 1, size=[input_batch_size, input_seq_size, input_hidden_size], dtype=torch.float32)

    # instantiate feed forward attention reducer
    transformer_stack = TransformerStack(
        input_hidden_size,
        input_seq_size,
        input_hidden_size,
        heads,
        output_size,
        num_layers,
    )

    result = transformer_stack(current_inputs)

    # ensure returned tensor is the correct shape
    assert list(result.shape) == [input_batch_size, input_seq_size, input_hidden_size]

    # check for parameter updating if fully connected layer is present
    target = torch.randn(result.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        transformer_stack,
        (current_inputs,),
        target,
    )
    assert upc == tpc, f"Some parameters not updated.  These parameters not updated: {not_updated}"
