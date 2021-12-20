import pytest
import torch

from ludwig.constants import HIDDEN, LOGITS
from ludwig.decoders.sequence_tagger import SequenceTaggerDecoder


@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_sequence_tagger(use_attention, use_bias):
    batch_size = 20
    combiner_output_state_size = 100
    vocab_size = 150
    max_sequence_length = 30
    decoder_inputs = {HIDDEN: torch.rand(batch_size, max_sequence_length, combiner_output_state_size)}
    tagger_decoder = SequenceTaggerDecoder(
        combiner_output_state_size, vocab_size, max_sequence_length, use_attention=use_attention, use_bias=use_bias
    )

    outputs = tagger_decoder(decoder_inputs)

    assert outputs[LOGITS].size() == torch.Size([batch_size, max_sequence_length, vocab_size])
