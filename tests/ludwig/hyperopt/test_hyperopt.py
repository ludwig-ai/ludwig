import pytest

from ludwig.constants import INPUT_FEATURES, NAME, OUTPUT_FEATURES, TYPE
from ludwig.hyperopt.utils import log_warning_if_all_grid_type_parameters, substitute_parameters
from ludwig.schema.model_config import ModelConfig

BASE_CONFIG = {
    INPUT_FEATURES: [{NAME: "title", TYPE: "text"}],
    OUTPUT_FEATURES: [{NAME: "summary", TYPE: "text"}],
}


def _get_config():
    return {
        "input_features": [{"name": "Date received", "type": "category"}],
        "output_features": [{"name": "Product", "type": "category"}],
        "hyperopt": {
            "goal": "minimize",
            "metric": "loss",
            "executor": {
                "type": "ray",
                "scheduler": {
                    "type": "async_hyperband",
                    "max_t": 3600,
                    "time_attr": "time_total_s",
                    "grace_period": 72,
                    "reduction_factor": 5,
                },
                "num_samples": 10,
                "time_budget_s": 3600,
                "cpu_resources_per_trial": 1,
            },
            "parameters": {"trainer.learning_rate": {"space": "choice", "categories": [0.005, 0.01, 0.02, 0.025]}},
            "search_alg": {"type": "hyperopt", "random_state_seed": 42},
            "output_feature": "Product",
        },
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


def test_grid_search_more_than_one_sample():
    """Test logs a user warning indicating that duplicate trials will be created because all of the parameters in
    the search space are of type grid_search and the number of samples is greater than 1."""
    with pytest.warns(RuntimeWarning):
        log_warning_if_all_grid_type_parameters(
            {
                "parameters": {
                    "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.005, 0.1]},
                    "defaults.text.encoder.type": {"space": "grid_search", "values": ["parallel_cnn", "stacked_cnn"]},
                },
                "executor": {"num_samples": 2},
            }
        )


def test_hyperopt_config_gbm():
    """This test was added due to a schema validation error when hyperopting GBMs:

    ```jsonschema.exceptions.ValidationError: Additional properties are not allowed ('epochs' was unexpected)```
    """
    config = _get_config()
    config["model_type"] = "gbm"

    # Config should not raise an exception
    ModelConfig.from_dict(config)


@pytest.mark.parametrize(
    "parameters, expected_num_samples",
    [
        (
            {
                "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.005, 0.1]},
                "defaults.category.encoder.type": {"space": "grid_search", "values": ["dense", "sparse"]},
            },
            1,
        ),
        (
            {
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.0001,
                    "upper": 0.01,
                },
                "defaults.category.encoder.type": {"space": "grid_search", "values": ["dense", "sparse"]},
            },
            1,
        ),
        (
            {
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.0001,
                    "upper": 0.01,
                },
            },
            10,
        ),
    ],
    ids=["all_grid_search", "mixed", "no_grid_search"],
)
def test_default_num_samples(parameters, expected_num_samples):
    """This test ensures that the default number of samples is set correctly when the user does not specify the
    number of samples in the hyperopt config."""
    config = _get_config()

    # Override to set num_samples to None so we can test inference logic
    config["hyperopt"]["executor"]["num_samples"] = None
    config["hyperopt"]["parameters"] = parameters

    processed_config = ModelConfig.from_dict(config).to_dict()

    assert processed_config["hyperopt"]["executor"]["num_samples"] == expected_num_samples
