import os

import pytest

from ludwig.constants import COLUMN, DECODER, DEFAULTS, INPUT_FEATURES, NAME, OUTPUT_FEATURES, SEQUENCE, TEXT, TYPE
from tests.integration_tests.utils import (
    category_feature,
    create_data_set_to_use,
    generate_data,
    RAY_BACKEND_CONFIG,
    sequence_feature,
    text_feature,
    train_with_backend,
)


# TODO: Add sequence feature and additional deocders
@pytest.mark.parametrize("feature_type", [TEXT])
@pytest.mark.parametrize("use_default", [True, False])
@pytest.mark.parametrize("decoder_type", ["generator"])
@pytest.mark.distributed
def test_sequence_decoder_predictions(tmpdir, csv_filename, ray_cluster_2cpu, feature_type, use_default, decoder_type):
    """Test that sequence decoders return the correct successfully predict."""
    input_features = [category_feature()]
    output_feature = (
        sequence_feature(output_feature=True) if feature_type == "sequence" else text_feature(output_feature=True)
    )
    dataset_path = generate_data(
        input_features=input_features,
        output_features=[output_feature],
        filename=os.path.join(tmpdir, csv_filename),
    )
    dataset_path = create_data_set_to_use("csv", dataset_path)

    config = {INPUT_FEATURES: input_features}

    # Ensure that the decoder outputs the correct predictions through both the default and feature-specific configs.
    if use_default:
        config[OUTPUT_FEATURES] = [{TYPE: feature_type, NAME: output_feature[NAME], COLUMN: output_feature[COLUMN]}]
        config[DEFAULTS] = {feature_type: {DECODER: {TYPE: decoder_type}}}
    else:
        config[OUTPUT_FEATURES] = [output_feature]

    # Test with decoder in output feature config
    train_with_backend(RAY_BACKEND_CONFIG, config=config, dataset=dataset_path)
