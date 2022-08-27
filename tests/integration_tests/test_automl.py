import os

import pytest

from ludwig.automl.automl import train_with_config
from ludwig.constants import TRAINER
from tests.integration_tests.utils import category_feature, generate_data, number_feature


@pytest.mark.distributed
@pytest.mark.parametrize("use_scheduler", [True, False])
def test_train_with_config(use_scheduler, ray_cluster_2cpu, tmpdir):
    input_features = [
        number_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3})]
    dataset = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    sampler = {
        "type": "ray",
        "num_samples": 2,
    }

    if use_scheduler:
        sampler["scheduler"] = {
            "type": "async_hyperband",
            "max_t": 200,
            "time_attr": "time_total_s",
            "grace_period": 72,
            "reduction_factor": 5,
        }

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
            "sampler": sampler,
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
    expected_early_stop = -1 if use_scheduler else 5
    assert best_model.config[TRAINER]["early_stop"] == expected_early_stop
