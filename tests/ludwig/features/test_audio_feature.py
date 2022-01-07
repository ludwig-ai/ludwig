from random import choice
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Dict

import pytest
import torch

from ludwig.features.audio_feature import AudioInputFeature

BATCH_SIZE = 2
SEQ_SIZE = 20
AUDIO_W_SIZE = 16
DEFAULT_FC_SIZE = 256

CHARS = ascii_uppercase + ascii_lowercase + digits
VOCAB = ["".join(choice(CHARS) for _ in range(2)) for _ in range(256)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def audio_config():
    return {
        "name": "audio_feature",
        "type": "audio",
        "preprocessing": {
            "audio_feature": {
                "type": "fbank",
                "window_length_in_s": 0.04,
                "window_shift_in_s": 0.02,
                "num_filter_bands": 80,
            },
            "audio_file_length_limit_in_s": 3.0,
        },
        "should_embed": False,
        "vocab": VOCAB,
        "max_sequence_length": SEQ_SIZE,
        "embedding_size": AUDIO_W_SIZE,
    }


@pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"])
def test_audio_input_feature(audio_config: Dict, encoder: str) -> None:
    audio_config.update({"encoder": encoder})
    audio_input_feature = AudioInputFeature(audio_config).to(DEVICE)
    audio_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE, AUDIO_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = audio_input_feature(audio_tensor)
    assert encoder_output["encoder_output"].shape[1:] == audio_input_feature.output_shape()
