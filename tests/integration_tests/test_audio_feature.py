import pytest
import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.features.audio_feature import AudioInputFeature
from ludwig.schema.features.audio_feature import AudioInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from tests.integration_tests.utils import audio_feature

BATCH_SIZE = 2
SEQ_SIZE = 20
AUDIO_W_SIZE = 16
DEFAULT_OUTPUT_SIZE = 256


@pytest.mark.parametrize("enc_encoder", ["stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"])
def test_audio_feature(enc_encoder):
    # synthetic audio tensor
    audio_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE, AUDIO_W_SIZE], dtype=torch.float32)

    # generate audio feature config
    audio_feature_config = audio_feature(
        folder=".", encoder={"type": enc_encoder, "max_sequence_length": SEQ_SIZE, "embedding_size": AUDIO_W_SIZE}
    )

    # instantiate audio input feature object
    audio_feature_config, _ = load_config_with_kwargs(AudioInputFeatureConfig, audio_feature_config)
    audio_input_feature = AudioInputFeature(audio_feature_config)

    # pass synthetic audio tensor through the audio input feature
    encoder_output = audio_input_feature(audio_tensor)

    # confirm correctness of the the audio encoder output
    assert isinstance(encoder_output, dict)
    assert ENCODER_OUTPUT in encoder_output
    assert isinstance(encoder_output[ENCODER_OUTPUT], torch.Tensor)
    if enc_encoder == "passthrough":
        assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, SEQ_SIZE, AUDIO_W_SIZE)
    else:
        assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, DEFAULT_OUTPUT_SIZE)
