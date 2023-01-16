import os
import tempfile

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from tests.integration_tests.utils import binary_feature, generate_data, number_feature, text_feature


def run_test_suite(config, dataset, backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        model = LudwigModel(config, backend=backend)
        _, _, output_dir = model.train(dataset=dataset, output_directory=tmpdir)

        model_dir = os.path.join(output_dir, "model")
        loaded_model = LudwigModel.load(model_dir, backend=backend)
        loaded_model.predict(dataset=dataset)


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
        text_feature(encoder={"type": "distilbert"}, preprocessing={"cache_encoder_embeddings": True}),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}
    run_test_suite(config, dataset, backend)


def test_cache_error(tmpdir):
    input_features = [
        number_feature(),
        text_feature(encoder={"type": "parallel_cnn"}, preprocessing={"cache_encoder_embeddings": True}),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}

    with pytest.raises(RuntimeError):
        run_test_suite(config, dataset, "local")
