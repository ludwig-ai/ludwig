import os
import random
import string

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, NAME, PROC_COLUMN, TRAINER
from ludwig.data.concatenate_datasets import concatenate_df
from tests.integration_tests.utils import (
    audio_feature,
    binary_feature,
    category_feature,
    generate_data,
    image_feature,
    init_backend,
    LocalTestBackend,
    number_feature,
    sequence_feature,
)

NUM_EXAMPLES = 10


@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.distributed
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


@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.distributed
def test_with_split(backend, csv_filename, tmpdir):
    num_examples = NUM_EXAMPLES
    train_set_size = int(num_examples * 0.8)
    val_set_size = int(num_examples * 0.1)
    test_set_size = int(num_examples * 0.1)

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=num_examples
    )
    data_df = pd.read_csv(data_csv)
    data_df["split"] = [0] * train_set_size + [1] * val_set_size + [2] * test_set_size
    data_df.to_csv(data_csv, index=False)
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 2,
        },
    }

    with init_backend(backend):
        model = LudwigModel(config, backend=backend)
        train_set, val_set, test_set, _ = model.preprocess(
            data_csv,
            skip_save_processed_input=False,
        )
        assert len(train_set) == train_set_size
        assert len(val_set) == val_set_size
        assert len(test_set) == test_set_size


@pytest.mark.parametrize("feature_fn", [image_feature, audio_feature])
@pytest.mark.distributed
def test_dask_known_divisions(feature_fn, csv_filename, tmpdir):
    import dask.dataframe as dd

    num_examples = NUM_EXAMPLES

    input_features = [feature_fn(os.path.join(tmpdir, "generated_output"))]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=num_examples
    )
    data_df = dd.from_pandas(pd.read_csv(data_csv), npartitions=1)
    assert data_df.known_divisions

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 2,
        },
    }

    backend = "ray"
    with init_backend(backend):
        model = LudwigModel(config, backend=backend)
        train_set, val_set, test_set, _ = model.preprocess(
            data_df,
            skip_save_processed_input=False,
        )


@pytest.mark.parametrize("generate_images_as_numpy", [False, True])
def test_read_image_from_path(tmpdir, csv_filename, generate_images_as_numpy):
    input_features = [image_feature(os.path.join(tmpdir, "generated_output"), save_as_numpy=generate_images_as_numpy)]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {"epochs": 2},
    }

    model = LudwigModel(config)
    model.preprocess(
        data_csv,
        skip_save_processed_input=False,
    )


def test_read_image_from_numpy_array(tmpdir, csv_filename):
    input_features = [image_feature(os.path.join(tmpdir, "generated_output"))]
    output_features = [category_feature(vocab_size=5, reduce_input="sum")]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    df = pd.read_csv(data_csv)
    processed_df_rows = []

    for _, row in df.iterrows():
        processed_df_rows.append(
            {
                input_features[0][NAME]: np.array(Image.open(row[input_features[0][NAME]])),
                output_features[0][NAME]: row[output_features[0][NAME]],
            }
        )

    df_with_images_as_numpy_arrays = pd.DataFrame(processed_df_rows)

    model = LudwigModel(config)
    model.preprocess(
        df_with_images_as_numpy_arrays,
        skip_save_processed_input=False,
    )


def test_number_feature_wrong_dtype(csv_filename, tmpdir):
    """Tests that a number feature with all string values is treated as having missing values by default."""
    data_csv_path = os.path.join(tmpdir, csv_filename)

    num_feat = number_feature()
    input_features = [num_feat]
    output_features = [binary_feature()]
    config = {"input_features": input_features, "output_features": output_features}

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # convert numbers to random strings
    def random_string():
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for _ in range(10))

    df[num_feat[COLUMN]] = df[num_feat[COLUMN]].apply(lambda _: random_string())

    # run preprocessing
    backend = LocalTestBackend()
    ludwig_model = LudwigModel(config, backend=backend)
    train_ds, val_ds, test_ds, _ = ludwig_model.preprocess(dataset=df)

    concatenated_df = concatenate_df(train_ds.to_df(), val_ds.to_df(), test_ds.to_df(), backend)

    # check that train_ds had invalid values replaced with the missing value
    assert len(concatenated_df) == len(df)
    assert np.all(concatenated_df[num_feat[PROC_COLUMN]] == 0.0)
