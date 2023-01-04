import os

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    number_feature,
)


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
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}

    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(dataset=training_data_csv_path)
