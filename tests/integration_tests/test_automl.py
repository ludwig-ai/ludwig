import os

import pytest

from ludwig.automl.automl import train_with_config
from ludwig.constants import TRAINER
from tests.integration_tests.utils import category_feature, generate_data, number_feature

CONFIG = {
    "input_features": [
        {"name": "PClass", "type": "category"},
        {"name": "Sex", "type": "category"},
        {"name": "Age", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "SibSp", "type": "number"},
        {"name": "Parch", "type": "number"},
        {"name": "Fare", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "Embarked", "type": "category"},
    ],
    "output_features": [{"name": "Survived", "type": "binary"}],
    "trainer": {"epochs": 2},
}


@pytest.mark.distributed
def test_train_with_config(ray_cluster_2cpu, tmpdir):
    input_features = [
        number_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3})]
    dataset = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {"epochs": 2},
        "hyperopt": {
            "search_alg": {
                "type": "hyperopt",
                "random_state_seed": 42,
            },
            "executor": {
                "type": "ray",
                "time_budget_s": 200,
                "cpu_resources_per_trial": 1,
            },
            "sampler": {
                "type": "ray",
                "num_samples": 2,
                "scheduler": {
                    "type": "async_hyperband",
                    "max_t": 200,
                    "time_attr": "time_total_s",
                    "grace_period": 72,
                    "reduction_factor": 5,
                },
            },
            "parameters": {
                "trainer.batch_size": {
                    "space": "choice",
                    "categories": [64, 128, 256],
                },
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
            },
        },
    }

    outdir = os.path.join(tmpdir, "output")
    results = train_with_config(dataset, config, output_directory=outdir)
    best_model = results.best_model

    # Early stopping in the rendered config needs to be disabled to allow the hyperopt scheduler to
    # manage trial lifecycle.
    assert best_model.config[TRAINER]["early_stop"] == -1
