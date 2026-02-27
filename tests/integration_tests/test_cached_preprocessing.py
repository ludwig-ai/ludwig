import os

import numpy as np
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import MODEL_ECD, PREPROCESSING, PROC_COLUMN, TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    number_feature,
    run_test_suite,
    text_feature,
)


def _onehot_encoding_config(tmpdir):
    input_features = [
        number_feature(),
        category_feature(encoder={"type": "onehot"}),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"train_steps": 1}}
    return config, dataset


def test_onehot_encoding(tmpdir):
    config, dataset = _onehot_encoding_config(tmpdir)
    run_test_suite(config, dataset, "local")


@pytest.mark.slow
@pytest.mark.distributed
def test_onehot_encoding_ray(tmpdir, ray_cluster_2cpu):
    config, dataset = _onehot_encoding_config(tmpdir)
    run_test_suite(config, dataset, "ray")


def _hf_text_embedding_config(tmpdir):
    input_features = [
        number_feature(),
        text_feature(
            encoder={
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "hf-internal-testing/tiny-bert-for-token-classification",
            },
            preprocessing={"cache_encoder_embeddings": True},
        ),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"train_steps": 1}}
    return config, dataset


def test_hf_text_embedding(tmpdir):
    config, dataset = _hf_text_embedding_config(tmpdir)
    run_test_suite(config, dataset, "local")


@pytest.mark.slow
@pytest.mark.distributed
def test_hf_text_embedding_ray(tmpdir, ray_cluster_2cpu):
    config, dataset = _hf_text_embedding_config(tmpdir)
    run_test_suite(config, dataset, "ray")


@pytest.mark.parametrize("cache_encoder_embeddings", [True, False, None])
def test_onehot_encoding_preprocessing(cache_encoder_embeddings, tmpdir):
    vocab_size = 5
    input_features = [
        category_feature(encoder={"type": "onehot", "vocab_size": vocab_size}),
        number_feature(),
    ]
    output_features = [binary_feature()]

    if cache_encoder_embeddings is not None:
        if PREPROCESSING not in input_features[0]:
            input_features[0][PREPROCESSING] = {}
        input_features[0][PREPROCESSING]["cache_encoder_embeddings"] = cache_encoder_embeddings

    # Need sufficiently high number of examples to ensure at least one of each category type appears
    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    num_examples = 100
    dataset_fp = generate_data(input_features, output_features, data_csv_path, num_examples)
    config = {
        "model_type": MODEL_ECD,
        "input_features": input_features,
        "output_features": output_features,
    }

    # Run preprocessing
    ludwig_model = LudwigModel(config, backend="local")
    proc_dataset = ludwig_model.preprocess(training_set=dataset_fp)

    # Check preprocessed output
    proc_df = ludwig_model.backend.df_engine.compute(proc_dataset.training_set.to_df())
    proc_col = input_features[0][PROC_COLUMN]
    proc_series = proc_df[proc_col]

    # ECD will not cache embeddings by default, but will if set to `cache_encoder_embeddings=true`
    expected_cache_encoder_embeddings = cache_encoder_embeddings or False
    if expected_cache_encoder_embeddings:
        assert proc_series.values.dtype == "object"
        data = np.stack(proc_series.values)
        assert data.shape == (num_examples, vocab_size)

        # Only one element in each row should be 1
        assert all(x == 1 for x in data.sum(axis=1))
    else:
        assert proc_series.values.dtype == "int8"
        data = proc_series.to_numpy()
        assert data.shape == (num_examples,)


def test_hf_text_embedding_tied(tmpdir):
    input_features = [
        text_feature(
            encoder={
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "hf-internal-testing/tiny-bert-for-token-classification",
            },
            preprocessing={"cache_encoder_embeddings": True},
        ),
        text_feature(
            encoder={
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "hf-internal-testing/tiny-bert-for-token-classification",
            },
            preprocessing={"cache_encoder_embeddings": True},
        ),
    ]
    input_features[1]["tied"] = input_features[0]["name"]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}
    run_test_suite(config, dataset, "local")
