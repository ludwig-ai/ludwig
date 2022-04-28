from random import choice
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Dict

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.signal.windows import get_window
import torch
import torchaudio

from ludwig.features.audio_feature import AudioInputFeature
import ludwig.utils.audio_utils as audio_utils

BATCH_SIZE = 2
SEQ_SIZE = 20
AUDIO_W_SIZE = 16

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
    assert encoder_output["encoder_output"].shape[1:] == audio_input_feature.output_shape


def test_lfilter(audio_config: Dict) -> None:
    raw_audio_tensor = torch.randn([1, 10000], dtype=torch.float64)
    emphasize_value = 0.97
    filter_window = np.asarray([1.0, -emphasize_value])
    pre_emphasized_data_expected = lfilter(filter_window, 1, raw_audio_tensor)

    filter_window = torch.tensor(filter_window, dtype=torch.float64)
    pre_emphasized_data = torchaudio.functional.lfilter(
        raw_audio_tensor, torch.tensor([1, 0], dtype=torch.float64), filter_window, clamp=False
    ).to(dtype=torch.float32)

    assert np.allclose(pre_emphasized_data_expected, pre_emphasized_data.numpy())


@pytest.mark.parametrize("window_type", ["bartlett", "blackman", "hamming", "hann", "kaiser"])
def test_get_window(window_type):
    window_length_in_samp = 10
    window_expected = get_window(window_type, window_length_in_samp, fftbins=False)

    window = audio_utils.get_window(window_type, window_length_in_samp)

    print(window_type)
    print(window_expected)
    print(window)
    assert np.allclose(window_expected, window.numpy())
