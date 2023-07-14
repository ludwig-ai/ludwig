import os
from random import choice
from string import ascii_lowercase, ascii_uppercase, digits

import pandas as pd
import pytest
import torch

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import BFILL, ENCODER_OUTPUT, PROC_COLUMN
from ludwig.features.audio_feature import AudioFeatureMixin, AudioInputFeature
from ludwig.schema.features.audio_feature import AudioInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.utils import audio_feature, category_feature, generate_data

BATCH_SIZE = 2
SEQ_SIZE = 20
AUDIO_W_SIZE = 16

CHARS = ascii_uppercase + ascii_lowercase + digits
VOCAB = ["".join(choice(CHARS) for _ in range(2)) for _ in range(256)]
DEVICE = get_torch_device()


@pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"])
def test_audio_input_feature(encoder: str) -> None:
    audio_config = {
        "name": "audio_feature",
        "type": "audio",
        "preprocessing": {
            "type": "fbank",
            "window_length_in_s": 0.04,
            "window_shift_in_s": 0.02,
            "num_filter_bands": 80,
            "audio_file_length_limit_in_s": 3.0,
        },
        "encoder": {
            "type": encoder,
            "should_embed": False,
            "vocab": VOCAB,
            "max_sequence_length": SEQ_SIZE,
            "embedding_size": AUDIO_W_SIZE,
        },
    }

    audio_config, _ = load_config_with_kwargs(AudioInputFeatureConfig, audio_config)
    audio_input_feature = AudioInputFeature(audio_config)

    audio_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE, AUDIO_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = audio_input_feature(audio_tensor)
    assert encoder_output[ENCODER_OUTPUT].shape[1:] == audio_input_feature.output_shape


@pytest.mark.parametrize("feature_type", ["raw", "stft", "stft_phase", "group_delay", "fbank"])
def test_add_feature_data(feature_type, tmpdir):
    preprocessing_params = {
        "audio_file_length_limit_in_s": 3.0,
        "missing_value_strategy": BFILL,
        "in_memory": True,
        "padding_value": 0,
        "norm": "per_file",
        "type": feature_type,
        "window_length_in_s": 0.04,
        "window_shift_in_s": 0.02,
        "num_fft_points": None,
        "window_type": "hamming",
        "num_filter_bands": 80,
    }
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")
    audio_feature_config = audio_feature(audio_dest_folder, preprocessing=preprocessing_params)
    data_df_path = generate_data(
        [audio_feature_config],
        [category_feature(vocab_size=5, reduce_input="sum")],
        os.path.join(tmpdir, "data.csv"),
        num_examples=10,
    )
    data_df = pd.read_csv(data_df_path)
    metadata = {
        audio_feature_config["name"]: AudioFeatureMixin.get_feature_meta(
            data_df[audio_feature_config["name"]], preprocessing_params, LOCAL_BACKEND, True
        )
    }

    proc_df = {}
    AudioFeatureMixin.add_feature_data(
        feature_config=audio_feature_config,
        input_df=data_df,
        proc_df=proc_df,
        metadata=metadata,
        preprocessing_parameters=preprocessing_params,
        backend=LOCAL_BACKEND,
        skip_save_processed_input=False,
    )

    assert len(proc_df[audio_feature_config[PROC_COLUMN]]) == 10
