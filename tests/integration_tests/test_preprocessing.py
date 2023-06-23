import contextlib
import copy
import logging
import os
import random
import string
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from transformers import AutoTokenizer

import ludwig
from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import (
    BASE_MODEL,
    BATCH_SIZE,
    COLUMN,
    DECODER,
    EPOCHS,
    FULL,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_LLM,
    MODEL_TYPE,
    NAME,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PROC_COLUMN,
    PROMPT,
    SPLIT,
    TRAINER,
    TYPE,
)
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.preprocessing import handle_features_with_prompt_config, preprocess_for_prediction
from ludwig.schema.llms.prompt import PromptConfig
from ludwig.schema.model_types.base import ModelConfig
from tests.integration_tests.utils import (
    assert_preprocessed_dataset_shape_and_dtype_for_feature,
    audio_feature,
    binary_feature,
    category_feature,
    generate_data,
    generate_data_as_dataframe,
    image_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    text_feature,
)

NUM_EXAMPLES = 20


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_sample_ratio(backend, tmpdir, ray_cluster_2cpu):
    num_examples = 100
    sample_ratio = 0.25

    input_features = [sequence_feature(encoder={"reduce_output": "sum"}), audio_feature(folder=tmpdir)]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=num_examples
    )
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            EPOCHS: 2,
        },
        PREPROCESSING: {"sample_ratio": sample_ratio},
    }

    model = LudwigModel(config, backend=backend)
    train_set, val_set, test_set, training_set_metadata = model.preprocess(
        data_csv,
        skip_save_processed_input=True,
    )

    sample_size = num_examples * sample_ratio
    count = len(train_set) + len(val_set) + len(test_set)
    assert sample_size == count

    # Check that sample ratio is disabled when doing preprocessing for prediction
    dataset, _ = preprocess_for_prediction(
        model.config_obj.to_dict(),
        dataset=data_csv,
        training_set_metadata=training_set_metadata,
        split=FULL,
        include_outputs=True,
        backend=model.backend,
    )
    assert "sample_ratio" in model.config_obj.preprocessing.to_dict()
    assert len(dataset) == num_examples


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_sample_ratio_deterministic(backend, tmpdir, ray_cluster_2cpu):
    """Ensures that the sampled dataset is the same when using a random seed.

    model.preprocess returns a PandasPandasDataset object when using local backend, and returns a RayDataset object when
    using the Ray backend.
    """
    num_examples = 100
    sample_ratio = 0.3

    input_features = [binary_feature()]
    output_features = [category_feature()]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=num_examples
    )

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        PREPROCESSING: {"sample_ratio": sample_ratio},
    }

    model1 = LudwigModel(config, backend=backend)
    train_set_1, val_set_1, test_set_1, _ = model1.preprocess(
        data_csv,
        skip_save_processed_input=True,
    )

    model2 = LudwigModel(config, backend=backend)
    train_set_2, val_set_2, test_set_2, _ = model2.preprocess(
        data_csv,
        skip_save_processed_input=True,
    )

    sample_size = num_examples * sample_ratio

    # Ensure sizes are the same
    assert sample_size == len(train_set_1) + len(val_set_1) + len(test_set_1)
    assert sample_size == len(train_set_2) + len(val_set_2) + len(test_set_2)

    # Ensure actual rows are the same
    if backend == "local":
        assert train_set_1.to_df().equals(train_set_2.to_df())
        assert val_set_1.to_df().equals(val_set_2.to_df())
        assert test_set_1.to_df().equals(test_set_2.to_df())
    else:
        assert train_set_1.to_df().compute().equals(train_set_2.to_df().compute())
        assert val_set_1.to_df().compute().equals(val_set_2.to_df().compute())
        assert test_set_1.to_df().compute().equals(test_set_2.to_df().compute())


def test_strip_whitespace_category(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    input_features = [binary_feature()]
    cat_feat = category_feature(decoder={"vocab_size": 3})
    output_features = [cat_feat]
    backend = LocalTestBackend()
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # prefix with whitespace
    df[cat_feat[COLUMN]] = df[cat_feat[COLUMN]].apply(lambda s: " " + s)

    # run preprocessing
    ludwig_model = LudwigModel(config, backend=backend)
    train_ds, _, _, metadata = ludwig_model.preprocess(dataset=df)

    # expect values containing whitespaces to be properly mapped to vocab_size unique values
    assert len(np.unique(train_ds.dataset[cat_feat[PROC_COLUMN]])) == cat_feat[DECODER]["vocab_size"]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_with_split(backend, csv_filename, tmpdir, ray_cluster_2cpu):
    num_examples = NUM_EXAMPLES
    train_set_size = int(num_examples * 0.8)
    val_set_size = int(num_examples * 0.1)
    test_set_size = int(num_examples * 0.1)

    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=num_examples
    )
    data_df = pd.read_csv(data_csv)
    data_df[SPLIT] = [0] * train_set_size + [1] * val_set_size + [2] * test_set_size
    data_df.to_csv(data_csv, index=False)
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            EPOCHS: 2,
        },
        PREPROCESSING: {SPLIT: {TYPE: "fixed", COLUMN: SPLIT}},
    }

    model = LudwigModel(config, backend=backend)
    train_set, val_set, test_set, _ = model.preprocess(
        data_csv,
        skip_save_processed_input=False,
    )
    assert len(train_set) == train_set_size
    assert len(val_set) == val_set_size
    assert len(test_set) == test_set_size


@pytest.mark.distributed
@pytest.mark.parametrize("feature_fn", [image_feature, audio_feature])
def test_dask_known_divisions(feature_fn, csv_filename, tmpdir, ray_cluster_2cpu):
    import dask.dataframe as dd

    input_features = [feature_fn(os.path.join(tmpdir, "generated_output"))]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]
    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=100)
    data_df = dd.from_pandas(pd.read_csv(data_csv), npartitions=2)
    assert data_df.known_divisions

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            EPOCHS: 2,
        },
    }

    backend = "ray"
    model = LudwigModel(config, backend=backend)
    train_set, val_set, test_set, _ = model.preprocess(
        data_df,
        skip_save_processed_input=False,
    )


@pytest.mark.distributed
def test_drop_empty_partitions(csv_filename, tmpdir, ray_cluster_2cpu):
    import dask.dataframe as dd

    input_features = [image_feature(os.path.join(tmpdir, "generated_output"))]
    output_features = [category_feature(vocab_size=5, reduce_input="sum", output_feature=True)]

    # num_examples and npartitions set such that each post-split DataFrame has >1 samples, but empty partitions.
    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=25)
    data_df = dd.from_pandas(pd.read_csv(data_csv), npartitions=10)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            EPOCHS: 2,
        },
    }

    backend = "ray"
    model = LudwigModel(config, backend=backend)
    train_set, val_set, test_set, _ = model.preprocess(
        data_df,
        skip_save_processed_input=True,
    )
    for dataset in [train_set, val_set, test_set]:
        df = dataset.ds.to_dask()
        for partition in df.partitions:
            assert len(partition) > 0, "empty partitions found in dataset"


@pytest.mark.parametrize("generate_images_as_numpy", [False, True])
def test_read_image_from_path(tmpdir, csv_filename, generate_images_as_numpy):
    input_features = [image_feature(os.path.join(tmpdir, "generated_output"), save_as_numpy=generate_images_as_numpy)]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]
    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {EPOCHS: 2},
    }

    model = LudwigModel(config)
    model.preprocess(
        data_csv,
        skip_save_processed_input=False,
    )


def test_read_image_from_numpy_array(tmpdir, csv_filename):
    input_features = [image_feature(os.path.join(tmpdir, "generated_output"))]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {EPOCHS: 2, BATCH_SIZE: 128},
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


def test_read_image_failure_default_image(monkeypatch, tmpdir, csv_filename):
    """Tests that the default image used when an image cannot be read has the correct properties."""

    def mock_read_binary_files(self, column, map_fn, file_size):
        """Mock read_binary_files to return None (failed image read) to test error handling."""
        return column.map(lambda x: None)

    monkeypatch.setattr(ludwig.backend.base.LocalPreprocessingMixin, "read_binary_files", mock_read_binary_files)

    image_feature_config = image_feature(os.path.join(tmpdir, "generated_output"))
    input_features = [image_feature_config]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {EPOCHS: 2, BATCH_SIZE: 128},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES, nan_percent=0.2
    )

    model = LudwigModel(config)
    preprocessed_dataset = model.preprocess(data_csv)
    training_set_metadata = preprocessed_dataset.training_set_metadata

    preprocessing = training_set_metadata[input_features[0][NAME]][PREPROCESSING]
    expected_shape = (preprocessing["num_channels"], preprocessing["height"], preprocessing["width"])
    expected_dtype = np.float32

    assert_preprocessed_dataset_shape_and_dtype_for_feature(
        image_feature_config[NAME], preprocessed_dataset, model.config_obj, expected_dtype, expected_shape
    )


def test_number_feature_wrong_dtype(csv_filename, tmpdir):
    """Tests that a number feature with all string values is treated as having missing values by default."""
    data_csv_path = os.path.join(tmpdir, csv_filename)

    num_feat = number_feature()
    input_features = [num_feat]
    output_features = [binary_feature()]
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

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


@pytest.mark.parametrize(
    "max_len, sequence_length, max_sequence_length, sequence_length_expected",
    [
        # Case 1: infer from the dataset, max_sequence_length is larger than the largest sequence length.
        # Expected: max_sequence_length is ignored, and the sequence length is dataset+2 (include start/stop tokens).
        (10, None, 15, 12),
        # Case 2: infer from the dataset, max_sequence_length is smaller than the largest sequence length.
        # Expected: max_sequence_length is used, and the sequence length is max_sequence_length.
        (10, None, 8, 8),
        # Case 3: infer from the dataset, max_sequence_length is not set.
        # Expected: max_sequence_length is ignored, and the sequence length is dataset+2 (include start/stop tokens).
        (10, None, None, 12),
        # Case 4: set sequence_length explicitly and it is larger than the dataset.
        # Expected: sequence_length is used, and the sequence length is sequence_length.
        (10, 15, 20, 15),
        # Case 5: set sequence_length explicitly and it is smaller than the dataset.
        # Expected: sequence_length is used, and the sequence length is sequence_length.
        (10, 8, 20, 8),
    ],
)
@pytest.mark.parametrize(
    "feature_type",
    [
        sequence_feature,
        text_feature,
    ],
)
def test_seq_features_max_sequence_length(
    csv_filename, tmpdir, feature_type, max_len, sequence_length, max_sequence_length, sequence_length_expected
):
    """Tests that a sequence feature has the correct max_sequence_length in metadata and prepocessed data."""
    feat = feature_type(
        encoder={"max_len": max_len},
        preprocessing={"sequence_length": sequence_length, "max_sequence_length": max_sequence_length},
    )
    input_features = [feat]
    output_features = [binary_feature()]
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

    data_csv_path = os.path.join(tmpdir, csv_filename)
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    class CheckTrainingSetMetadataCallback(Callback):
        def on_preprocess_end(self, proc_training_set, proc_validation_set, proc_test_set, training_set_metadata):
            assert training_set_metadata[feat[NAME]]["max_sequence_length"] == sequence_length_expected

    backend = LocalTestBackend()
    ludwig_model = LudwigModel(config, backend=backend, callbacks=[CheckTrainingSetMetadataCallback()])
    train_ds, val_ds, test_ds, _ = ludwig_model.preprocess(dataset=df)

    all_df = concatenate_df(train_ds.to_df(), val_ds.to_df(), test_ds.to_df(), backend)
    proc_column_name = feat[PROC_COLUMN]
    assert all(len(x) == sequence_length_expected for x in all_df[proc_column_name])


def test_column_feature_type_mismatch_fill():
    """Tests that we are able to fill missing values even in columns where the column dtype and desired feature
    dtype do not match."""
    cat_feat = category_feature()
    bin_feat = binary_feature()
    input_features = [cat_feat]
    output_features = [bin_feat]
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

    # Construct dataframe with int-like column representing a categorical feature
    df = pd.DataFrame(
        {
            cat_feat[NAME]: pd.Series(pd.array([None] + [1] * 24, dtype=pd.Int64Dtype())),
            bin_feat[NAME]: pd.Series([True] * 25),
        }
    )

    # run preprocessing
    backend = LocalTestBackend()
    ludwig_model = LudwigModel(config, backend=backend)
    train_ds, val_ds, test_ds, _ = ludwig_model.preprocess(dataset=df)


@pytest.mark.parametrize("format", ["file", "df"])
def test_presplit_override(format, tmpdir):
    """Tests that provising a pre-split file or dataframe overrides the user's split config."""
    num_feat = number_feature(normalization=None)
    input_features = [num_feat, sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=25)
    data_df = pd.read_csv(data_csv)

    # Set the feature value equal to an ordinal index so we can ensure the splits are identical before and after
    # preprocessing.
    data_df[num_feat[COLUMN]] = data_df.index

    train_df = data_df[:15]
    val_df = data_df[15:20]
    test_df = data_df[20:]

    train_data = train_df
    val_data = val_df
    test_data = test_df

    if format == "file":
        train_data = os.path.join(tmpdir, "train.csv")
        val_data = os.path.join(tmpdir, "val.csv")
        test_data = os.path.join(tmpdir, "test.csv")

        train_df.to_csv(train_data)
        val_df.to_csv(val_data)
        test_df.to_csv(test_data)

    data_df.to_csv(data_csv, index=False)
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            EPOCHS: 2,
        },
        PREPROCESSING: {SPLIT: {TYPE: "random"}},
    }

    model = LudwigModel(config, backend=LocalTestBackend())
    train_set, val_set, test_set, _ = model.preprocess(
        training_set=train_data, validation_set=val_data, test_set=test_data
    )

    assert len(train_set) == len(train_df)
    assert len(val_set) == len(val_df)
    assert len(test_set) == len(test_df)

    assert np.all(train_set.to_df()[num_feat[PROC_COLUMN]].values == train_df[num_feat[COLUMN]].values)
    assert np.all(val_set.to_df()[num_feat[PROC_COLUMN]].values == val_df[num_feat[COLUMN]].values)
    assert np.all(test_set.to_df()[num_feat[PROC_COLUMN]].values == test_df[num_feat[COLUMN]].values)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_empty_training_set_error(backend, tmpdir, ray_cluster_2cpu):
    """Tests that an error is raised if one or more of the splits is empty after preprocessing."""
    data_csv_path = os.path.join(tmpdir, "data.csv")

    out_feat = binary_feature()
    input_features = [number_feature()]
    output_features = [out_feat]
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # Convert all the output features rows to null. Because the default missing value strategy is to drop empty output
    # rows, this will result in the dataset being empty after preprocessing.
    df[out_feat[COLUMN]] = None

    ludwig_model = LudwigModel(config, backend=backend)
    with pytest.raises(RuntimeError, match="Training data is empty following preprocessing"):
        ludwig_model.preprocess(dataset=df)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_in_memory_dataset_size(backend, tmpdir, ray_cluster_2cpu):
    data_csv_path = os.path.join(tmpdir, "data.csv")

    out_feat = binary_feature()
    input_features = [number_feature()]
    output_features = [out_feat]
    config = {INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features}

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    ludwig_model = LudwigModel(config, backend=backend)
    training_dataset, validation_dataset, test_dataset, _ = ludwig_model.preprocess(dataset=df)

    assert training_dataset.in_memory_size_bytes > 0
    assert validation_dataset.in_memory_size_bytes > 0
    assert test_dataset.in_memory_size_bytes > 0


@pytest.mark.parametrize(
    "binary_as_input, expected_preprocessing, missing_value_strategy",
    [
        pytest.param(
            True,
            {
                "missing_value_strategy": "fill_with_true",
                "fill_value": None,
                "computed_fill_value": ">50K",
                "fallback_true_label": ">50K",
            },
            "fill_with_true",
            id="binary_as_input_1",
        ),
        pytest.param(
            True,
            {
                "missing_value_strategy": "fill_with_false",
                "fill_value": None,
                "computed_fill_value": "<=50K",
                "fallback_true_label": ">50K",
            },
            "fill_with_false",
            id="binary_as_input_2",
        ),
        pytest.param(
            False,
            {
                "missing_value_strategy": "drop_row",
                "fill_value": None,
                "computed_fill_value": None,
                "fallback_true_label": ">50K",
            },
            "drop_row",
            id="binary_as_output",
        ),
    ],
)
def test_non_conventional_bool_with_fallback(binary_as_input, expected_preprocessing, missing_value_strategy, tmpdir):
    # Specify a non-conventional boolean feature with a fallback true label.
    bin_feature = binary_feature(
        bool2str=["<=50K", ">50K"],
        preprocessing={"fallback_true_label": ">50K", "missing_value_strategy": missing_value_strategy},
    )

    # Generate data with the non-conventional boolean feature.
    if binary_as_input:
        input_features = [bin_feature]
        output_features = [number_feature()]
    else:
        input_features = [number_feature()]
        output_features = [bin_feature]
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {EPOCHS: 2, BATCH_SIZE: 128},
    }

    data_csv_path = os.path.join(tmpdir, "data.csv")
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # Preprocess the data.
    ludwig_model = LudwigModel(config)
    _, _, _, training_set_metadata = ludwig_model.preprocess(dataset=df)

    # Check that true/false labels are set correctly.
    assert training_set_metadata[bin_feature[NAME]] == {
        "str2bool": {"<=50K": False, ">50K": True},
        "bool2str": ["<=50K", ">50K"],
        "fallback_true_label": ">50K",
        PREPROCESSING: expected_preprocessing,
    }


@pytest.mark.parametrize(
    "binary_as_input", [pytest.param(True, id="binary_as_input"), pytest.param(False, id="binary_as_output")]
)
def test_non_conventional_bool_without_fallback_logs_warning(binary_as_input, caplog, tmpdir):
    # Specify a non-conventional boolean feature without a fallback true label.
    bin_feature = binary_feature(bool2str=["<=50K", ">50K"], preprocessing={"fallback_true_label": None})

    # Generate data with the non-conventional boolean feature.
    if binary_as_input:
        input_features = [bin_feature]
        output_features = [number_feature()]
    else:
        input_features = [number_feature()]
        output_features = [bin_feature]
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {EPOCHS: 2, BATCH_SIZE: 128},
    }

    data_csv_path = os.path.join(tmpdir, "data.csv")
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = pd.read_csv(training_data_csv_path)

    # Preprocess the data.
    with caplog.at_level(logging.WARN, logger="ludwig.features.binary_feature"):
        ludwig_model = LudwigModel(config)
        ludwig_model.preprocess(dataset=df)

    # Check that a warning is logged.
    assert "unconventional boolean value" in caplog.text


@pytest.mark.parametrize("feature_type", ["input_feature", "output_feature"], ids=["input_feature", "output_feature"])
def test_category_feature_vocab_size_1(feature_type, tmpdir) -> None:
    data_csv_path = os.path.join(tmpdir, "data.csv")

    input_feature = [category_feature(encoder={"vocab_size": 1})]
    output_feature = [binary_feature()]

    if feature_type == "output_feature":
        input_feature = output_feature
        output_feature = [category_feature(decoder={"vocab_size": 1})]

    config = {INPUT_FEATURES: input_feature, OUTPUT_FEATURES: output_feature, "training": {EPOCHS: 1}}

    training_data_csv_path = generate_data(config[INPUT_FEATURES], config[OUTPUT_FEATURES], data_csv_path)

    ludwig_model = LudwigModel(config)
    with pytest.raises(RuntimeError) if feature_type == "output_feature" else contextlib.nullcontext():
        ludwig_model.train(dataset=training_data_csv_path)


@pytest.mark.parametrize("use_pretrained", [False, True], ids=["false", "true"])
def test_vit_encoder_different_dimension_image(tmpdir, csv_filename, use_pretrained: bool):
    input_features = [
        image_feature(
            os.path.join(tmpdir, "generated_output"),
            preprocessing={"in_memory": True, "height": 224, "width": 206, "num_channels": 3},
            encoder={TYPE: "_vit_legacy", "use_pretrained": use_pretrained},
        )
    ]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {"train_steps": 1},
    }

    model = LudwigModel(config)

    # Failure happens post preprocessing but before training during the ECD model creation phase
    # so make sure the model can be created properly and training can proceed
    model.train(dataset=data_csv)


@pytest.mark.skip(
    reason=(
        "Broken against torch nightly: "
        "https://github.com/ludwig-ai/ludwig/actions/runs/4918126111/jobs/8784071603?pr=3388."
    )
)
def test_image_encoder_torchvision_different_num_channels(tmpdir, csv_filename):
    input_features = [
        image_feature(
            os.path.join(tmpdir, "generated_output"),
            preprocessing={"in_memory": True, "height": 224, "width": 206, "num_channels": 1},
            encoder={TYPE: "efficientnet"},
        )
    ]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {"train_steps": 1},
    }

    model = LudwigModel(config)

    # Failure happens post preprocessing but before training during the ECD model creation phase
    # so make sure the model can be created properly and training can proceed
    model.train(dataset=data_csv)


@pytest.mark.parametrize(
    "df_engine",
    [
        pytest.param("pandas", id="pandas"),
        pytest.param("dask", id="dask", marks=pytest.mark.distributed),
    ],
)
def test_fill_with_mode_different_df_engine(tmpdir, csv_filename, df_engine, ray_cluster_2cpu):
    config = {
        INPUT_FEATURES: [category_feature(preprocessing={"missing_value_strategy": "fill_with_mode"})],
        OUTPUT_FEATURES: [binary_feature()],
    }

    training_data_csv_path = generate_data(
        config[INPUT_FEATURES], config[OUTPUT_FEATURES], os.path.join(tmpdir, csv_filename)
    )

    df = pd.read_csv(training_data_csv_path)

    if df_engine == "dask":
        import dask.dataframe as dd

        df = dd.from_pandas(df, npartitions=1)

        # Only support Dask on Ray backend
        config["backend"] = {TYPE: "ray"}

    ludwig_model = LudwigModel(config)
    ludwig_model.preprocess(dataset=df)


template_task_sample = """
Instruction: {__task__}
###
Examples:
###
Input: foo bar
Output: true
###
Input: baz quc
Output: false
###
Input: {__sample__}
Output:
"""

task = "predict the output feature. Return only values in {true, false}"

template_multi_col = """
You are a helpful chatbot. USER: {__sample__}: {country}, {year:.2f} ASSISTANT:
"""

expected_task_sample = """instruction: predict the output feature. return only values in {true, false}
###
examples:
###
input: foo bar
output: true
###
input: baz quc
output: false
###
input:"""


@pytest.mark.llm
@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize("model_type", [MODEL_ECD, MODEL_LLM])
@pytest.mark.parametrize(
    "input_features,expected",
    [
        (
            [
                text_feature(
                    preprocessing={
                        PROMPT: {"task": task, "template": template_task_sample},
                        "max_sequence_length": 512,
                    }
                )
            ],
            expected_task_sample,
        ),
        (
            [
                text_feature(preprocessing={PROMPT: {"template": template_multi_col}}),
                category_feature(name="country"),
                number_feature(name="year"),
            ],
            ("you are a helpful chatbot. user: "),
        ),
    ],
    ids=["task_sample", "multi_col"],
)
def test_prompt_template(input_features, expected, model_type, backend, tmpdir, ray_cluster_2cpu):
    """Tests that prompt template is correctly applied to inputs."""
    input_features = copy.deepcopy(input_features)

    output_features = [category_feature()]
    data_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=25)

    data_df = pd.read_csv(data_csv)
    raw_values = [data_df[input_features[i][COLUMN]].values.tolist() for i in range(len(input_features))]

    # Only use the first input featuere (text) and discard the others, which are only used for data gen
    input_features = input_features[:1]
    config = {
        MODEL_TYPE: model_type,
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model_name = "hf-internal-testing/tiny-random-OPTModel"
    if model_type == MODEL_LLM:
        # For LLMs, specify the prompt at the top level
        config[BASE_MODEL] = model_name
        config[PROMPT] = input_features[0][PREPROCESSING][PROMPT]
        del config[INPUT_FEATURES][0][PREPROCESSING][PROMPT]
        config[INPUT_FEATURES][0]["encoder"] = {TYPE: "passthrough"}
    else:
        config[INPUT_FEATURES][0]["encoder"] = {
            TYPE: "auto_transformer",
            "pretrained_model_name_or_path": model_name,
        }

    model = LudwigModel(config, backend=backend)
    train_set, _, _, _ = model.preprocess(
        training_set=data_csv,
        skip_save_processed_input=True,
        output_directory=os.path.join(tmpdir, "processed"),
    )

    train_df = model.backend.df_engine.compute(train_set.to_df())
    encoded_values = train_df[input_features[0][PROC_COLUMN]].values.tolist()

    assert all(len(v) == len(encoded_values) for v in raw_values)

    for i, encoded in enumerate(encoded_values):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        decoded = tokenizer.decode(encoded)
        assert expected in decoded, f"decoded: '{decoded}' does not contain expected: {expected}"

        for raw_col_values in raw_values:
            v = raw_col_values[i]
            if isinstance(v, float):
                # Test formatting in parametrize uses 2 decimal places of precision
                raw_text = f"{v:.2f}"
            else:
                raw_text = str(v).lower()
            assert raw_text in decoded, f"'{raw_text}' not in '{decoded}'"


@pytest.mark.llm
@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize(
    "retrieval_kwargs",
    [
        pytest.param({"type": "random", "k": 2}, id="random_retrieval"),
        # TODO: find a smaller model for testing
        pytest.param({"type": "semantic", "model_name": "paraphrase-MiniLM-L3-v2", "k": 2}, id="semantic_retrieval"),
    ],
)
def test_handle_features_with_few_shot_prompt_config(backend, retrieval_kwargs, ray_cluster_2cpu):
    prompt_config = PromptConfig.from_dict(
        {
            "task": (
                "Given the sample input, complete this sentence by replacing XXXX: "
                "The label is XXXX. Choose one value in this list: [1, 2, 3]."
            ),
            "retrieval": retrieval_kwargs,
        }
    ).to_dict()  # convert back-and-forth to validate and add defaults

    input_features = [
        text_feature(
            encoder={TYPE: "passthrough"},
        )
    ]
    output_features = [
        category_feature(
            output_feature=True,
            decoder={TYPE: "category_extractor"},
        )
    ]
    input_feature_name = input_features[0][NAME]
    output_feature_name = output_features[0][NAME]

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "gpt2",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        PROMPT: prompt_config,
    }
    config = ModelConfig.from_dict(config).to_dict()

    df = generate_data_as_dataframe(input_features, output_features, 10, with_split=True)  # retrieval needs fixed split
    if backend == "ray":
        import dask.dataframe as dd

        df = dd.from_pandas(df, npartitions=2)

    split_col = df[SPLIT]
    feature_configs = config[INPUT_FEATURES] + config[OUTPUT_FEATURES]

    if backend == "local":
        context = mock.patch(
            "ludwig.models.retrieval.SemanticRetrieval._encode",
            side_effect=lambda row_strs, _: np.random.rand(len(row_strs), 16).astype(np.float32),
        )
    else:
        # TODO: figure out how to get mocks to work with Ray backend
        context = contextlib.nullcontext()

    with context:
        backend = initialize_backend(backend)
        dataset_cols = handle_features_with_prompt_config(
            config,
            df,
            feature_configs,
            backend=backend,
            split_col=split_col,
        )

        assert len(dataset_cols) == 1
        assert input_feature_name in dataset_cols

        # Inspect the generated prompts
        col = backend.df_engine.compute(dataset_cols[input_feature_name])
        for prompt in col:
            # input_feature_name and output_feature_name should be in the prompt because
            # labeled samples are provided by the context
            assert input_feature_name in prompt
            assert output_feature_name in prompt


@pytest.mark.llm
@pytest.mark.parametrize("backend", ["local", "ray"])
def test_handle_features_with_prompt_config_multi_col(backend, ray_cluster_2cpu):
    df = pd.DataFrame(
        [
            {
                "instruction": "Name this province",
                "country": "Canada",
                "year": 1871,
                "answer": "British Columbia",
            },
            {
                "instruction": "Name this city",
                "country": "France",
                "year": 1789,
                "answer": "Paris",
            },
            {
                "instruction": "Name this country",
                "country": "UK",
                "year": 1057,
                "answer": "Wales",
            },
        ]
    )

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "gpt2",
        INPUT_FEATURES: [text_feature(name="question", encoder={TYPE: "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="answer")],
        PROMPT: {
            "template": "You are a helpful chatbot. USER: {instruction}: {country}, {year:.2f} ASSISTANT:",
        },
    }
    config = ModelConfig.from_dict(config).to_dict()

    if backend == "ray":
        import dask.dataframe as dd

        df = dd.from_pandas(df, npartitions=2)

    feature_configs = config[INPUT_FEATURES] + config[OUTPUT_FEATURES]

    backend = initialize_backend(backend)
    dataset_cols = handle_features_with_prompt_config(
        config,
        df,
        feature_configs,
        backend=backend,
        split_col=None,
    )

    assert len(dataset_cols) == 1
    assert "question" in dataset_cols

    col = backend.df_engine.compute(dataset_cols["question"])
    assert len(col) == 3
    assert col[0].startswith("You are a helpful chatbot. USER: Name this province: Canada, 1871.00 ASSISTANT:")
    assert col[1].startswith("You are a helpful chatbot. USER: Name this city: France, 1789.00 ASSISTANT:")
    assert col[2].startswith("You are a helpful chatbot. USER: Name this country: UK, 1057.00 ASSISTANT:")
