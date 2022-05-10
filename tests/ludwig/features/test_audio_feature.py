import os
from random import choice
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import COLUMN, NAME, PROC_COLUMN
from ludwig.features.audio_feature import AudioInputFeature, AudioFeatureMixin
from tests.integration_tests.utils import audio_feature, category_feature, generate_data


BATCH_SIZE = 2
SEQ_SIZE = 20
AUDIO_W_SIZE = 16

CHARS = ascii_uppercase + ascii_lowercase + digits
VOCAB = ["".join(choice(CHARS) for _ in range(2)) for _ in range(256)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"])
def test_audio_input_feature(encoder: str) -> None:
    audio_config = {
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
        "encoder": encoder,
    }
    audio_input_feature = AudioInputFeature(audio_config).to(DEVICE)
    audio_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE, AUDIO_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = audio_input_feature(audio_tensor)
    assert encoder_output["encoder_output"].shape[1:] == audio_input_feature.output_shape


# @pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"])
def test_add_feature_data(tmpdir):
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    # Single audio input, single category output
    input_features = [audio_feature(audio_dest_folder)]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "data.csv"))

    print(pd.read_csv(rel_path))

    # AudioFeatureMixin.add_feature_data(
    #     feature_config=num_feature,
    #     input_df=data_df,
    #     proc_df=proc_df,
    #     metadata={input_features[0]: b,
    #     preprocessing_parameters={"normalization": "zscore"},
    #     backend=LOCAL_BACKEND,
    #     skip_save_processed_input=False,
    # )
    # assert np.allclose(
    #     np.array(proc_df[num_feature[PROC_COLUMN]]), np.array([-1.26491106, -0.63245553, 0, 0.63245553, 1.26491106])
    # )

    # NumberFeatureMixin.add_feature_data(
    #     feature_config=num_feature,
    #     input_df=data_df,
    #     proc_df=proc_df,
    #     metadata={num_feature[NAME]: feature_2_meta},
    #     preprocessing_parameters={"normalization": "minmax"},
    #     backend=LOCAL_BACKEND,
    #     skip_save_processed_input=False,
    # )
    # assert np.allclose(np.array(proc_df[num_feature[PROC_COLUMN]]), np.array([0, 0.25, 0.5, 0.75, 1]))
