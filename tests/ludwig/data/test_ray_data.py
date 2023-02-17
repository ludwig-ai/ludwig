from unittest import mock

import pytest

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")  # noqa

from ludwig.data.dataset.ray import RayDatasetBatcher  # noqa

# Mark the entire module as distributed
pytestmark = pytest.mark.distributed


def test_async_reader_error():
    pipeline = mock.Mock()
    features = {
        "num1": {"name": "num1", "type": "number"},
        "bin1": {"name": "bin1", "type": "binary"},
    }
    training_set_metadata = {
        "num1": {},
        "bin1": {},
    }

    with pytest.raises(TypeError, match="'Mock' object is not iterable"):
        RayDatasetBatcher(
            dataset_epoch_iterator=iter([pipeline]),
            features=features,
            training_set_metadata=training_set_metadata,
            batch_size=64,
            samples_per_epoch=100,
            ignore_last=False,
        )
