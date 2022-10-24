from typing import Dict

import pytest
import torch

from ludwig.constants import ENCODER, TYPE
from ludwig.features.timeseries_feature import TimeseriesInputFeature
from ludwig.schema.features.timeseries_feature import TimeseriesInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.torch_utils import get_torch_device

SEQ_SIZE = 2
TIMESERIES_W_SIZE = 1
MAX_LEN = 7
EMBEDDING_SIZE = 5
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def timeseries_config():
    return {
        "name": "timeseries_12",
        "type": "timeseries",
        "encoder": {
            "max_len": MAX_LEN,
            "embedding_size": EMBEDDING_SIZE,
            "max_sequence_length": SEQ_SIZE,
            "output_size": 8,
            "state_size": 8,
            "num_filters": 8,
            "hidden_size": 8,
        },
    }


@pytest.mark.parametrize("encoder", ["rnn", "stacked_cnn", "parallel_cnn"])
def test_timeseries_input_feature(timeseries_config: Dict, encoder: str) -> None:
    timeseries_config[ENCODER][TYPE] = encoder

    timeseries_config, _ = load_config_with_kwargs(TimeseriesInputFeatureConfig, timeseries_config)
    timeseries_input_feature = TimeseriesInputFeature(timeseries_config).to(DEVICE)
    timeseries_tensor = torch.randn([SEQ_SIZE, TIMESERIES_W_SIZE], dtype=torch.float32).to(DEVICE)
    encoder_output = timeseries_input_feature(timeseries_tensor)
    assert encoder_output["encoder_output"].shape[1:] == timeseries_input_feature.output_shape
