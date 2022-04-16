import contextlib
import os

import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import category_feature, generate_data, ray_cluster, sequence_feature


@contextlib.contextmanager
def init_backend(backend: str):
    if backend == "local":
        with contextlib.nullcontext():
            yield
            return

    if backend == "ray":
        with ray_cluster():
            yield
            return

    raise ValueError(f"Unrecognized backend: {backend}")


@pytest.mark.parametrize("backend", ["local", "ray"])
def test_sample_ratio(backend, tmpdir):
    num_examples = 100
    sample_ratio = 0.25

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=num_examples
    )
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 2,
        },
        "preprocessing": {"sample_ratio": sample_ratio},
    }

    with init_backend(backend):
        model = LudwigModel(config, backend=backend)
        train_set, val_set, test_set, _ = model.preprocess(
            data_csv,
            skip_save_processed_input=True,
        )

        sample_size = num_examples * sample_ratio
        count = len(train_set) + len(val_set) + len(test_set)
        assert sample_size == count
