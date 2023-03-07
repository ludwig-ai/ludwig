import pytest

from ludwig.schema.hyperopt.search_algorithm import BaseSearchAlgorithmConfig, register_search_algorithm
from ludwig.schema.utils import ludwig_dataclass, ProtectedString


@register_search_algorithm("installed", dependencies=["ludwig"])
@ludwig_dataclass
class InstalledDependencySAConfig(BaseSearchAlgorithmConfig):
    type: str = ProtectedString("installed")


@register_search_algorithm("multi_installed", dependencies=["ludwig", "marshmallow"])
@ludwig_dataclass
class MultiInstalledDependencySAConfig(BaseSearchAlgorithmConfig):
    type: str = ProtectedString("multi_installed")


@register_search_algorithm("uninstalled", dependencies=["fictional_dependency"])
@ludwig_dataclass
class UninstalledDependencySAConfig(BaseSearchAlgorithmConfig):
    type: str = ProtectedString("uninstalled")


@register_search_algorithm("mixed", dependencies=["ludwig", "fictional_dependency"])
@ludwig_dataclass
class MixedDependencySAConfig(BaseSearchAlgorithmConfig):
    type: str = ProtectedString("uninstalled")


def test_dependency_check():
    """Test that the hyperopt search alg dependency check properly identifies missing dependencies.

    Most search algorithms supported by Ray Tune have additional dependencies that may not be installed. The schema
    records these dependencies and can be used to verify they are installed at run time.
    """
    # An empty dependency list should pass the check
    conf = BaseSearchAlgorithmConfig(type="random")
    assert conf.dependencies_installed()

    # An installed dependency should pass the check
    conf = InstalledDependencySAConfig()
    assert conf.dependencies_installed()

    # Multiple installed dependencies should pass the check
    conf = MultiInstalledDependencySAConfig()
    assert conf.dependencies_installed()

    # A missing dependency should fail the check
    conf = UninstalledDependencySAConfig()
    with pytest.raises(ImportError):
        assert conf.dependencies_installed()

    # A mix of installed and missing dependencies should fail the check
    conf = MixedDependencySAConfig()
    with pytest.raises(ImportError):
        assert conf.dependencies_installed()
