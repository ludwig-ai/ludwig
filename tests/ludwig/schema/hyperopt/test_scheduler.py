import pytest

from ludwig.schema.hyperopt.scheduler import BaseSchedulerConfig
from ludwig.schema.hyperopt.utils import register_scheduler_config, scheduler_config_registry
from ludwig.schema.utils import ludwig_dataclass, ProtectedString


@pytest.fixture(
    params=[  # Tuples of SA name, dependency list, whether it should raise an exception
        ("no_deps", None, False),
        ("installed", [("ludwig", "ludwig")], False),
        ("multiple_installed", [("ludwig", "ludwig"), ("marshmallow", "marshmallow")], False),
        ("not_installed", [("fake_dependency", "fake_dependency")], True),
        ("mixed_installed", [("fake_dependency", "fake_dependency"), ("ludwig", "ludwig")], True),
    ]
)
def dependency_check_config(request):
    key, deps, raises_exception = request.param

    @register_scheduler_config(key, dependencies=deps)
    @ludwig_dataclass
    class DependencyCheckConfig(BaseSchedulerConfig):
        type: str = ProtectedString(key)

    yield DependencyCheckConfig(), raises_exception
    del scheduler_config_registry[key]


def test_dependency_check(dependency_check_config):
    """Test that the hyperopt scheduler dependency check properly identifies missing dependencies.

    Some schedulers supported by Ray Tune have additional dependencies that may not be installed. The schema records
    these dependencies and can be used to verify they are installed at run time.
    """
    config, raises_exception = dependency_check_config
    if raises_exception:
        with pytest.raises(ImportError):
            config.dependencies_installed()
    else:
        assert config.dependencies_installed()
