import os
import tempfile

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from tests.integration_tests.utils import binary_feature, category_feature, generate_data, number_feature, text_feature


def run_test_suite(config, dataset, backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        ludwig_model = LudwigModel(config, backend=backend)
        ludwig_model.train(dataset=dataset, output_directory=tmpdir)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_onehot_encoding(tmpdir, backend, ray_cluster_2cpu):
    input_features = [
        number_feature(),
        category_feature(encoder={"type": "onehot"}),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}
    run_test_suite(config, dataset, backend)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_distilbert_embedding(tmpdir, backend, ray_cluster_2cpu):
    input_features = [
        number_feature(),
        text_feature(encoder={"type": "distilbert"}),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}
    run_test_suite(config, dataset, backend)
