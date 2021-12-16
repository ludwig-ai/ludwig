from typing import Dict

import pytest
import torch

from ludwig.features.timeseries_feature import TimeseriesInputFeature

SEQ_SIZE = 2
TIMESERIES_W_SIZE = 1
MAX_LEN = 7
EMBEDDING_SIZE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def timeseries_config():
    return {
        "name": "timeseries_12",
        "type": "timeseries",
        "max_len": MAX_LEN,
        "embedding_size": EMBEDDING_SIZE,
        "max_sequence_length": SEQ_SIZE,
        "fc_size": 8,
        "state_size": 8,
        "num_filters": 8,
        "hidden_size": 8,
    }


@pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn"])
def test_timeseries_input_feature(timeseries_config: Dict, encoder: str) -> None:
    timeseries_config.update({"encoder": encoder})
    timeseries_input_feature = TimeseriesInputFeature(timeseries_config)
    timeseries_tensor = torch.randn([SEQ_SIZE, TIMESERIES_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = timeseries_input_feature(timeseries_tensor)
    assert encoder_output["encoder_output"].shape[1:] == timeseries_input_feature.output_shape
