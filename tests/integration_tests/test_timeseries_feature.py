import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, ENCODER_OUTPUT, INPUT_FEATURES, OUTPUT_FEATURES
from ludwig.features.timeseries_feature import TimeseriesInputFeature
from ludwig.schema.features.timeseries_feature import TimeseriesInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from tests.integration_tests.utils import number_feature, timeseries_feature

BATCH_SIZE = 2
SEQ_SIZE = 10
DEFAULT_OUTPUT_SIZE = 4


@pytest.mark.parametrize(
    "enc_encoder", ["stacked_cnn", "parallel_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn", "passthrough"]
)
def test_timeseries_feature(enc_encoder):
    # synthetic time series tensor
    timeseries_tensor = torch.randn([BATCH_SIZE, SEQ_SIZE], dtype=torch.float32)

    # generate feature config
    timeseries_feature_config = timeseries_feature(
        encoder={
            "type": enc_encoder,
            "max_len": SEQ_SIZE,
            "fc_layers": [{"output_size": DEFAULT_OUTPUT_SIZE}],
            # simulated parameters determined by pre-processing
            "max_sequence_length": SEQ_SIZE,
        }
    )

    # instantiate input feature object
    timeseries_feature_config, _ = load_config_with_kwargs(TimeseriesInputFeatureConfig, timeseries_feature_config)
    timeseries_input_feature = TimeseriesInputFeature(timeseries_feature_config)

    # pass synthetic tensor through input feature
    encoder_output = timeseries_input_feature(timeseries_tensor)

    # confirm correctness of the encoder output
    assert isinstance(encoder_output, dict)
    assert ENCODER_OUTPUT in encoder_output
    assert isinstance(encoder_output[ENCODER_OUTPUT], torch.Tensor)
    if enc_encoder == "passthrough":
        assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, SEQ_SIZE, 1)
    else:
        assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, DEFAULT_OUTPUT_SIZE)


def test_timeseries_preprocessing_with_nan():
    config = {
        "input_features": [timeseries_feature(preprocessing={"padding_value": 42})],
        "output_features": [number_feature()],
    }

    # generate synthetic data
    data = {
        config[INPUT_FEATURES][0][COLUMN]: [
            "1.53 2.3 NaN 6.4 3 ",
            "1.53 2.3 2 ",
            "1.53 NaN 3 2 ",
        ],
        config[OUTPUT_FEATURES][0][COLUMN]: [1.0, 2.0, 3.0],
    }
    df = pd.DataFrame(data)

    model = LudwigModel(config)
    ds = model.preprocess(df)
    out_df = ds.training_set.to_df()

    assert len(out_df.columns) == len(df.columns)

    expected_df = pd.DataFrame(
        [
            [np.array([1.53, 2.3, 42.0, 6.4, 3.0]), 1.0],
            [np.array([1.53, 2.3, 2.0, 42.0, 42.0]), 2.0],
            [np.array([1.53, 42.0, 3.0, 2.0, 42.0]), 3.0],
        ],
        columns=out_df.columns.to_list(),
    )

    for row1, row2 in zip(out_df.values, expected_df.values):
        assert np.allclose(row1[0], row2[0])
        assert row1[1] == row2[1]
