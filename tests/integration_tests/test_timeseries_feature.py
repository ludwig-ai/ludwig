import pytest
import torch

from ludwig.features.timeseries_feature import TimeseriesInputFeature
from ludwig.schema.features.timeseries_feature import TimeseriesInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from tests.integration_tests.utils import timeseries_feature

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
    assert "encoder_output" in encoder_output
    assert isinstance(encoder_output["encoder_output"], torch.Tensor)
    if enc_encoder == "passthrough":
        assert encoder_output["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, 1)
    else:
        assert encoder_output["encoder_output"].shape == (BATCH_SIZE, DEFAULT_OUTPUT_SIZE)
