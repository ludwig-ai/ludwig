import pytest

from ludwig.backend import initialize_backend
from ludwig.constants import INPUT_FEATURES, NAME, OUTPUT_FEATURES, TYPE
from ludwig.hyperopt.run import update_or_set_max_concurrent_trials
from ludwig.hyperopt.utils import substitute_parameters

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


@pytest.mark.distributed
@pytest.mark.parametrize(
    "parameters, expected",
    [
        (
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
        ),
        (
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 2},
        ),
        (
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
            {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
        ),
    ],
    ids=["none", "auto", "1"],
)
def test_set_max_concurrent_trials(parameters, expected, ray_cluster_4cpu):
    backend = initialize_backend("ray")
    update_or_set_max_concurrent_trials(parameters, backend)
    assert parameters == expected
