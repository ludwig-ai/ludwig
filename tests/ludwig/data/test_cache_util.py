import uuid
from typing import List
from unittest import mock

import pytest

from ludwig.constants import INPUT_FEATURES, OUTPUT_FEATURES
from ludwig.data.cache.util import calculate_checksum
from ludwig.schema.model_types.base import ModelConfig
from ludwig.types import FeatureConfigDict, ModelConfigDict
from ludwig.utils.misc_utils import merge_dict


def _gen_config(input_features: List[FeatureConfigDict]) -> ModelConfigDict:
    return {INPUT_FEATURES: input_features, OUTPUT_FEATURES: [{"name": "out1", "type": "binary"}]}


@pytest.mark.parametrize(
    "input_features,diff,expected",
    [
        (
            [
                {
                    "name": "in1",
                    "type": "text",
                    "encoder": {"type": "parallel_cnn"},
                }
            ],
            [
                {
                    "encoder": {"type": "stacked_cnn"},
                }
            ],
            True,
        ),
        (
            [
                {
                    "name": "in1",
                    "type": "text",
                    "preprocessing": {"cache_encoder_embeddings": True},
                    "encoder": {"type": "bert"},
                }
            ],
            [
                {
                    "encoder": {"type": "distilbert"},
                }
            ],
            False,
        ),
    ],
)
def test_calculate_checksum(input_features: List[FeatureConfigDict], diff: List[FeatureConfigDict], expected: bool):
    config = _gen_config(input_features)

    diff_features = [merge_dict(f, df) for f, df in zip(input_features, diff)]
    diff_config = _gen_config(diff_features)

    mock_dataset = mock.Mock()
    mock_dataset.checksum = uuid.uuid4().hex

    assert (
        calculate_checksum(mock_dataset, ModelConfig.from_dict(config).to_dict())
        == calculate_checksum(mock_dataset, ModelConfig.from_dict(diff_config).to_dict())
    ) == expected
