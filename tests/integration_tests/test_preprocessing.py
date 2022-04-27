import contextlib
import os

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, PROC_COLUMN
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    ray_cluster,
    sequence_feature,
)


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


def test_strip_whitespace_category(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    input_features = [binary_feature()]
    cat_feat = category_feature(vocab_size=3)
    output_features = [cat_feat]
    backend = LocalTestBackend()
    config = {"input_features": input_features, "output_features": output_features}

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # prefix with whitespace
    df[cat_feat[COLUMN]] = df[cat_feat[COLUMN]].apply(lambda s: " " + s)

    # run preprocessing
    ludwig_model = LudwigModel(config, backend=backend)
    train_ds, _, _, metadata = ludwig_model.preprocess(dataset=df)

    # expect values containing whitespaces to be properly mapped to vocab_size unique values
    assert len(np.unique(train_ds.dataset[cat_feat[PROC_COLUMN]])) == cat_feat["vocab_size"]
