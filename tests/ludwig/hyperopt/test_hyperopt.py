import pytest

from ludwig.backend import initialize_backend
from ludwig.constants import INPUT_FEATURES, NAME, OUTPUT_FEATURES, TYPE
from ludwig.hyperopt.utils import substitute_parameters, update_or_set_max_concurrent_trials

BASE_CONFIG = {
    INPUT_FEATURES: [{NAME: "title", TYPE: "text"}],
    OUTPUT_FEATURES: [{NAME: "summary", TYPE: "text"}],
}


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (
            {
                "combiner.type": "tabnet",
                "combiner.fc_layers": [{"output_size": 64}, {"output_size": 32}],
                "trainer.learning_rate": 0.1,
                "trainer.batch_size": 256,
            },
            {
                **BASE_CONFIG,
                "combiner": {"type": "tabnet", "fc_layers": [{"output_size": 64}, {"output_size": 32}]},
                "trainer": {"learning_rate": 0.1, "batch_size": 256},
            },
        ),
        (
            {
                "title.encoder.type": "bert",
                "summary.decoder.reduce_input": "sum",
                "trainer.learning_rate": 0.1,
                "trainer.batch_size": 256,
            },
            {
                INPUT_FEATURES: [{NAME: "title", TYPE: "text", "encoder": {"type": "bert"}}],
                OUTPUT_FEATURES: [{NAME: "summary", TYPE: "text", "decoder": {"reduce_input": "sum"}}],
                "trainer": {"learning_rate": 0.1, "batch_size": 256},
            },
        ),
        (
            {
                ".": {
                    "combiner": {"type": "concat", "num_fc_layers": 2},
                    "trainer": {"learning_rate_scaling": "linear"},
                },
                "trainer.learning_rate": 0.1,
            },
            {
                **BASE_CONFIG,
                "combiner": {"type": "concat", "num_fc_layers": 2},
                "trainer": {"learning_rate_scaling": "linear", "learning_rate": 0.1},
            },
        ),
        (
            {
                ".": {
                    "combiner": {"type": "concat", "num_fc_layers": 2},
                    "trainer": {"learning_rate_scaling": "linear"},
                },
                "trainer": {
                    "learning_rate": 0.1,
                    "batch_size": 256,
                },
            },
            {
                **BASE_CONFIG,
                "combiner": {"type": "concat", "num_fc_layers": 2},
                "trainer": {"learning_rate_scaling": "linear", "learning_rate": 0.1, "batch_size": 256},
            },
        ),
    ],
    ids=["flat", "features", "nested", "multi-nested"],
)
def test_substitute_parameters(parameters, expected):
    actual_config = substitute_parameters(BASE_CONFIG, parameters)
    assert actual_config == expected


# @pytest.mark.parametrize("parameters, expected", [], ids=[])
# def test_get_total_trial_count(parameters, expected):
#     pass


@pytest.mark.distributed
@pytest.mark.parametrize(
    "parameters, expected",
    [
        (  # If max_concurrent_trials is none, it should not be set in the updated config
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
            },
        ),
        (  # If max_concurrent_trials is auto, set it to total_trials - 2 if num_samples == num_cpus
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 2},
            },
        ),
        (  # Even though num_samples is set to 4, this will actually result in 9 trials. This test checks if
            # we still correctly set max_concurrent_trials to 2
            {
                "parameters": {
                    "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.01, 0.1]},
                    "combiner.num_fc_layers": {"space": "grid_search", "values": [1, 2, 3]},
                },
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            },
            {
                "parameters": {
                    "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.01, 0.1]},
                    "combiner.num_fc_layers": {"space": "grid_search", "values": [1, 2, 3]},
                },
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 2},
            },
        ),
        (  # Ensure user config value (1) is respected if it is less than total possible trials (2)
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
            },
        ),
    ],
    ids=["none", "auto", "auto_with_large_num_trials", "1"],
)
def test_set_max_concurrent_trials(parameters, expected, ray_cluster_4cpu):
    backend = initialize_backend("ray")
    update_or_set_max_concurrent_trials(parameters, backend)
    assert parameters == expected
