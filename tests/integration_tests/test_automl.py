import os
from unittest import mock

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from tests.integration_tests.utils import category_feature, generate_data, number_feature

try:
    from ludwig.automl.automl import train_with_config
    from ludwig.hyperopt.execution import RayTuneExecutor
except ImportError:
    pass


@pytest.mark.distributed
@pytest.mark.parametrize("time_budget", [200, 1], ids=["high", "low"])
def test_train_with_config(time_budget, ray_cluster_2cpu, tmpdir):
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
                "time_budget_s": time_budget,
                "cpu_resources_per_trial": 1,
                "scheduler": {
                    "type": "async_hyperband",
                    "max_t": time_budget,
                    "time_attr": "time_total_s",
                    "grace_period": min(72, time_budget),
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

    fn = RayTuneExecutor._evaluate_best_model
    with mock.patch("ludwig.hyperopt.execution.RayTuneExecutor._evaluate_best_model") as mock_fn:
        # We need to check that _evaluate_best_model is called when the time_budget is low
        # as this code path should be triggered when the trial was early stopped
        mock_fn.side_effect = fn

        outdir = os.path.join(tmpdir, "output")
        results = train_with_config(dataset, config, output_directory=outdir)
        best_model = results.best_model

        if time_budget > 1:
            assert isinstance(best_model, LudwigModel)
            assert best_model.config[TRAINER]["early_stop"] == -1
            assert mock_fn.call_count == 0
        else:
            assert best_model is None
            assert mock_fn.call_count > 0
