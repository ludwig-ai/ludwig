import pytest
import torch

from ludwig.constants import HIDDEN, LOGITS
from ludwig.decoders.sequence_tagger import SequenceTaggerDecoder
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_sequence_tagger(use_attention, use_bias):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    batch_size = 20
    combiner_output_state_size = 100
    vocab_size = 150
    max_sequence_length = 30
    decoder_inputs = {HIDDEN: torch.rand(batch_size, max_sequence_length, combiner_output_state_size)}
    tagger_decoder = SequenceTaggerDecoder(
        combiner_output_state_size, vocab_size, max_sequence_length, use_attention=use_attention, use_bias=use_bias
    )

    outputs = tagger_decoder(decoder_inputs)

    assert outputs[LOGITS].size()[1:] == tagger_decoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs[LOGITS].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(tagger_decoder, (decoder_inputs,), target)
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"
