from itertools import repeat

import pytest

# Imported to populate the registry
import ludwig.schema.hyperopt.parameter  # noqa: F401
from ludwig.constants import (
    EXECUTOR,
    HYPEROPT,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    PARAMETERS,
    SCHEDULER,
    SEARCH_ALG,
    TYPE,
)
from ludwig.error import ConfigValidationError
from ludwig.schema.hyperopt import utils
from ludwig.schema.hyperopt.scheduler import BaseSchedulerConfig
from ludwig.schema.hyperopt.search_algorithm import BaseSearchAlgorithmConfig
from ludwig.schema.model_types.base import ModelConfig
from ludwig.schema.utils import ludwig_dataclass, ProtectedString
from tests.integration_tests.utils import binary_feature, text_feature

CONFIG_DEPENDENCY_PARAMS = {  # Tuples of type, dependency list, whether it should raise an exception
    "no_deps": {TYPE: "no_deps", "dependencies": None, "raises_exception": False},
    "installed": {TYPE: "installed", "dependencies": [("ludwig", "ludwig")], "raises_exception": False},
    "multiple_installed": {
        TYPE: "multiple_installed",
        "dependencies": [("ludwig", "ludwig"), ("marshmallow", "marshmallow")],
        "raises_exception": False,
    },
    "not_installed": {
        TYPE: "not_installed",
        "dependencies": [("fake_dependency", "fake_dependency")],
        "raises_exception": True,
    },
    "mixed_installed": {
        TYPE: "mixed_installed",
        "dependencies": [("ludwig", "ludwig"), ("fake_dependency", "fake_dependency")],
        "raises_exception": True,
    },
}


@pytest.fixture(scope="module")
def register_schedulers():
    for name, values in CONFIG_DEPENDENCY_PARAMS.items():

        @utils.register_scheduler_config(name, dependencies=values["dependencies"])
        @ludwig_dataclass
        class DependencyCheckConfig(BaseSchedulerConfig):
            type: str = ProtectedString(name)

    yield

    for name in CONFIG_DEPENDENCY_PARAMS.keys():
        del utils.scheduler_config_registry[name]
        del utils.scheduler_dependencies_registry[name]


@pytest.fixture(scope="module")
def register_search_algorithms():
    for name, values in CONFIG_DEPENDENCY_PARAMS.items():

        @utils.register_search_algorithm_config(name, dependencies=values["dependencies"])
        @ludwig_dataclass
        class DependencyCheckConfig(BaseSearchAlgorithmConfig):
            type: str = ProtectedString(name)

    yield

    for name in CONFIG_DEPENDENCY_PARAMS.keys():
        del utils.search_algorithm_config_registry[name]
        del utils.search_algorithm_random_state_field_registry[name]
        del utils.search_algorithm_dependencies_registry[name]


@pytest.mark.parametrize("name", CONFIG_DEPENDENCY_PARAMS.keys())
def test_check_scheduler_dependencies_installed(name, register_schedulers):
    scheduler_name = CONFIG_DEPENDENCY_PARAMS[name][TYPE]
    raises_exception = CONFIG_DEPENDENCY_PARAMS[name]["raises_exception"]

    config = {
        INPUT_FEATURES: [text_feature()],
        OUTPUT_FEATURES: [binary_feature()],
        HYPEROPT: {
            PARAMETERS: {"trainer.learning_rate": {"space": "choice", "categories": [0.0001, 0.001, 0.01, 0.1]}},
            EXECUTOR: {SCHEDULER: {TYPE: scheduler_name}},
        },
    }

    if raises_exception:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)
    else:
        ModelConfig.from_dict(config)


@pytest.mark.parametrize("name", CONFIG_DEPENDENCY_PARAMS.keys())
def test_check_search_algorithm_dependencies_installed(name, register_search_algorithms):
    search_algorithm_name = CONFIG_DEPENDENCY_PARAMS[name][TYPE]
    raises_exception = CONFIG_DEPENDENCY_PARAMS[name]["raises_exception"]

    config = {
        INPUT_FEATURES: [text_feature()],
        OUTPUT_FEATURES: [binary_feature()],
        HYPEROPT: {
            PARAMETERS: {"trainer.learning_rate": {"space": "choice", "categories": [0.0001, 0.001, 0.01, 0.1]}},
            SEARCH_ALG: {TYPE: search_algorithm_name},
        },
    }

    if raises_exception:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)
    else:
        ModelConfig.from_dict(config)


@pytest.mark.parametrize(
    "space,raises_exception",
    list(zip(utils.parameter_config_registry.keys(), repeat(False, len(utils.parameter_config_registry))))
    + [("fake_space", True)],
)
def test_parameter_type_check(space, raises_exception):
    """Test that the parameter type is a valid hyperparameter search space.

    This should only be valid until the search space schema is updated to validate spaces as config objects rather than
    dicts. That update is non-trivial, so to hold over until it is ready we cast the dicts to the corresponding
    parameter objects and validate as an aux check. The test covers every valid space and one invalid space.
    """
    config = {
        INPUT_FEATURES: [text_feature()],
        OUTPUT_FEATURES: [binary_feature()],
        HYPEROPT: {
            SEARCH_ALG: {TYPE: "random"},
            PARAMETERS: {
                "trainer.learning_rate": {
                    "space": space,
                    # "categories": [0.0001, 0.001, 0.01, 0.1]
                }
            },
        },
    }

    if not raises_exception:
        ModelConfig.from_dict(config)
    else:
        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config)


@pytest.mark.parametrize("referenced_parameter", ["", " ", ".", "foo.bar", "trainer.bar", "foo.learning_rate"])
def test_parameter_key_check(referenced_parameter):
    """Test that references to nonexistent config parameters raise validation errors.

    Hyperopt parameters reference the config parameters they search with `.` notation to access different subsections,
    e.g. `trainer.learning_rate`. These are added to the config as arbitrary strings, and an invalid reference should be
    considered a validation error since we will otherwise search over an unused space or defer the error to train time.
    """
    config = {
        INPUT_FEATURES: [text_feature()],
        OUTPUT_FEATURES: [binary_feature()],
        HYPEROPT: {
            SEARCH_ALG: {TYPE: "random"},
            PARAMETERS: {referenced_parameter: {"space": "choice", "categories": [0.0001, 0.001, 0.01, 0.1]}},
        },
    }

    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)
