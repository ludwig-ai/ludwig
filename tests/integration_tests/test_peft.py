import os

import pytest

from ludwig.constants import TRAINER
from tests.integration_tests.utils import binary_feature, generate_data, run_test_suite, text_feature


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_text_tuner_lora(tmpdir, backend, ray_cluster_2cpu):
    input_features = [
        text_feature(
            encoder={
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "hf-internal-testing/tiny-bert-for-token-classification",
                "trainable": True,
                "tuner": "lora",
            },
        ),
    ]
    output_features = [binary_feature()]

    data_csv_path = os.path.join(tmpdir, "dataset.csv")
    dataset = generate_data(input_features, output_features, data_csv_path)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}
    model = run_test_suite(config, dataset, backend)

    state_dict = model.model.state_dict()

    # check that at least one of the keys contains the word "lora_" denoting a lora parameter
    assert any("lora_" in key for key in state_dict.keys())
