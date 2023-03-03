from dataclasses import field
from importlib import import_module
from typing import Dict, List, Optional

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()
sa_dependencies_registry = Registry()


def register_search_algorithm(name: str):
    def wrap(cls):
        search_algorithm_registry[name] = cls
        sa_dependencies_registry[name] = cls.dependencies
        return cls

    return wrap


def get_search_algorithm_cls(name: str):
    return search_algorithm_registry[name]


@DeveloperAPI
@ludwig_dataclass
class BaseSearchAlgorithmConfig(schema_utils.BaseMarshmallowConfig):
    """Basic search algorithm settings."""

    type: str = schema_utils.StringOptions(
        options=list(search_algorithm_registry.keys()), default="hyperopt", allow_none=False
    )

    dependencies: Optional[List[str]] = schema_utils.List(
        list_type=str,
        default=(lambda x: list())(),
        description="List of the additional packages required for this search algorithm.",
    )

    def dependencies_installed(self) -> bool:
        """Some search algorithms require additional packages to be installed, check that they are available."""
        for package_name in sa_dependencies_registry[self.type]:
            try:
                import_module(package_name)
                return True
            except ImportError:
                raise ImportError(
                    f"Search algorithm {self.type} requires package {package_name}, however package is "
                    "not installed. Please refer to Ray Tune documentation for packages required for this "
                    "search algorithm."
                )


@DeveloperAPI
def SearchAlgorithmDataclassField(description: str = "", default: Dict = {"type": "variant_generator"}):
    class SearchAlgorithmMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return BaseSearchAlgorithmConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for scheduler: {value}, see SearchAlgorithmConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(BaseSearchAlgorithmConfig),
                "title": "scheduler",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: BaseSearchAlgorithmConfig.Schema().load(default)
    dump_default = BaseSearchAlgorithmConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": SearchAlgorithmMarshmallowField(
                allow_none=False,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=load_default,
    )


@DeveloperAPI
@register_search_algorithm("random")
@register_search_algorithm("variant_generator")
@ludwig_dataclass
class BasicVariantSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.StringOptions(options=["random", "variant_generator"], default="random", allow_none=False)

    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        )
    )

    max_concurrent: int = schema_utils.NonNegativeInteger(
        default=0, description="Maximum number of concurrently running trials. If 0 (default), no maximum is enforced."
    )

    constant_grid_search: bool = schema_utils.Boolean(
        default=False,
        description=(
            "If this is set to True, Ray Tune will first try to sample random values and keep them constant over grid "
            "search parameters. If this is set to False (default), Ray Tune will sample new random parameters in each "
            "grid search condition."
        ),
    )

    random_state: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Seed or numpy random generator to use for reproducible results. If None (default), will use the global "
            "numpy random generator (np.random). Please note that full reproducibility cannot be guaranteed in a "
            "distributed environment."
        ),
    )


@DeveloperAPI
@register_search_algorithm("ax")
@ludwig_dataclass
class AxSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("ax")

    dependencies: List[str] = ["ax-platform", "sqlalchemy"]

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            r"Parameters in the experiment search space. Required elements in the dictionaries are: \“name\” (name of "
            r"this parameter, string), \“type\” (type of the parameter: \“range\”, \“fixed\”, or \“choice\”, string), "
            r"\“bounds\” for range parameters (list of two values, lower bound first), \“values\” for choice "
            r"parameters (list of values), and \“value\” for fixed parameters (single value)."
        )
    )

    metric: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "Name of the metric used as objective in this experiment. This metric must be present in `raw_data` "
            "argument to `log_data`. This metric must also be present in the dict reported/returned by the Trainable. "
            "If `None` but a mode was passed, the `ray.tune.result.DEFAULT_METRIC` will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max", None],
        default=None,
        allow_none=True,
        description=(
            "One of `{min, max}`. Determines whether objective is minimizing or maximizing the metric attribute. "
            r"Defaults to \“max\”."
        ),
    )

    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good "
            "parameters you want to run first to help the algorithm make better suggestions for future parameters. "
            "Needs to be a list of dicts containing the configurations."
        )
    )

    parameter_constraints: Optional[List] = schema_utils.List(
        description=r"Parameter constraints, such as \“x3 >= x4\” or \“x3 + x4 >= 2\”."
    )

    outcome_constraints: Optional[List] = schema_utils.List(
        description=r"Outcome constraints of form \“metric_name >= bound\”, like \“m1 <= 3.\”"
    )


@DeveloperAPI
@register_search_algorithm("bayesopt")
@ludwig_dataclass
class BayesOptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bayesopt")

    dependencies: List[str] = ["bayesian-optimization"]

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "Continuous search space. Parameters will be sampled from this space which will be used to run trials"
        )
    )

    metric: Optional[str] = schema_utils.String(
        defualt=None,
        allow_none=True,
        description=(
            "The training result objective value attribute. If None but a mode was passed, the anonymous metric "
            "`_metric` will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        allow_none=True,
        description=(
            "One of `{min, max}`. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        )
    )

    utility_kwargs: Optional[Dict] = schema_utils.Dict(
        description=(
            "Parameters to define the utility function. The default value is a dictionary with three keys: "
            "- kind: ucb (Upper Confidence Bound) - kappa: 2.576 - xi: 0.0"
        )
    )

    random_state: int = schema_utils.Integer(default=42, description="Used to initialize BayesOpt.")

    random_search_steps: int = schema_utils.Integer(
        default=10,
        description=(
            "Number of initial random searches. This is necessary to avoid initial local overfitting of "
            "the Bayesian process."
        ),
    )

    verbose: int = schema_utils.IntegerOptions(
        options=[0, 1, 2], default=0, description="The level of verbosity. `0` is least verbose, `2` is most verbose."
    )

    patience: int = schema_utils.NonNegativeInteger(
        default=5, description="Number of epochs to wait for a change in the top models."
    )

    skip_duplicate: bool = schema_utils.Boolean(
        default=True,
        description=(
            "If False, the optimizer will allow duplicate points to be registered. This behavior may be desired in "
            "high noise situations where repeatedly probing the same point will give different answers. In other "
            "situations, the acquisition may occasionaly generate a duplicate point."
        ),
    )


@DeveloperAPI
@register_search_algorithm("blendsearch")
@ludwig_dataclass
class BlendsearchSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("blendsearch")

    dependencies: List[str] = ["flaml"]


@DeveloperAPI
@register_search_algorithm("bohb")
@ludwig_dataclass
class BOHBSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bohb")

    dependencies: List[str] = ["hpbandster", "ConfigSpace"]

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "Continuous ConfigSpace search space. Parameters will be sampled from this space which will be used "
            "to run trials."
        )
    )

    bohb_config: Optional[Dict] = schema_utils.Dict(description="configuration for HpBandSter BOHB algorithm")

    metric: Optional[str] = schema_utils.String(
        description=(
            "The training result objective value attribute. If None but a mode was passed, the anonymous metric "
            "`_metric` will be used per default."
        )
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        allow_none=True,
        description=(
            "One of `{min, max}`. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )
    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        )
    )

    seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Optional random seed to initialize the random number generator. Setting this should lead to identical "
            "initial configurations at each run."
        ),
    )

    max_concurrent: int = schema_utils.Integer(
        default=0,
        description=(
            "Number of maximum concurrent trials. If this Searcher is used in a `ConcurrencyLimiter`, the "
            "`max_concurrent` value passed to it will override the value passed here. Set to <= 0 for no limit on "
            "concurrency."
        ),
    )


@DeveloperAPI
@register_search_algorithm("cfo")
@ludwig_dataclass
class CFOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("cfo")

    dependencies: List[str] = ["flaml", "cfo"]


@DeveloperAPI
@register_search_algorithm("dragonfly")
@ludwig_dataclass
class DragonflySAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("dragonfly")

    dependencies: List[str] = ["dragonfly-opt"]

    optimizer: Optional[str] = schema_utils.StringOptions(
        options=["random", "bandit", "genetic"],
        default=None,
        allow_none=True,
        description=(
            "Optimizer provided from dragonfly. Choose an optimiser that extends `BlackboxOptimiser`. If this is a "
            "string, `domain` must be set and `optimizer` must be one of [random, bandit, genetic]."
        ),
    )

    domain: Optional[str] = schema_utils.StringOptions(
        options=["cartesian", "euclidean"],
        default=None,
        allow_none=True,
        description=(
            "Optional domain. Should only be set if you don't pass an optimizer as the `optimizer` argument. Has to "
            "be one of [cartesian, euclidean]."
        ),
    )

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Search space. Should only be set if you don't pass an optimizer as the `optimizer` argument. Defines the "
            "search space and requires a `domain` to be set. Can be automatically converted from the `param_space` "
            "dict passed to `tune.Tuner()`."
        )
    )

    metric: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "The training result objective value attribute. If None but a mode was passed, the anonymous metric "
            "`_metric` will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        allow_none=True,
        description=(
            "One of `{min, max}`. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        )
    )

    evaluated_rewards: Optional[List] = schema_utils.List(
        description=(
            "If you have previously evaluated the parameters passed in as points_to_evaluate you can avoid re-running "
            "those trials by passing in the reward attributes as a list so the optimiser can be told the results "
            "without needing to re-compute the trial. Must be the same length as `points_to_evaluate`."
        )
    )

    random_state_seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Seed for reproducible results. Defaults to None. Please note that setting this to a value will change "
            "global random state for `numpy` on initalization and loading from checkpoint."
        ),
    )


@DeveloperAPI
@register_search_algorithm("hebo")
@ludwig_dataclass
class HEBOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("hebo")

    dependencies: List[str] = ["hebo"]

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=("A dict mapping parameter names to Tune search spaces or a HEBO DesignSpace object.")
    )

    metric: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "The training result objective value attribute. If None but a mode was passed, the anonymous metric "
            "`_metric` will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        allow_none=True,
        description=(
            "One of `{min, max}`. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    points_to_evaluate: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        )
    )

    evaluated_rewards: Optional[List] = schema_utils.List(
        description=(
            "If you have previously evaluated the parameters passed in as points_to_evaluate you can avoid re-running "
            "those trials by passing in the reward attributes as a list so the optimiser can be told the results "
            "without needing to re-compute the trial. Must be the same length as `points_to_evaluate`."
        )
    )

    random_state_seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Seed for reproducible results. Defaults to None. Please note that setting this to a value will change "
            "global random state for `numpy` on initalization and loading from checkpoint."
        ),
    )

    max_concurrent: int = schema_utils.NonNegativeInteger(
        default=8,
        description=(
            "Number of maximum concurrent trials. If this Searcher is used in a `ConcurrencyLimiter`, the "
            "`max_concurrent` value passed to it will override the value passed here."
        ),
    )


@DeveloperAPI
@register_search_algorithm("hyperopt")
@ludwig_dataclass
class HyperoptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("hyperopt")


@DeveloperAPI
@register_search_algorithm("nevergrad")
@ludwig_dataclass
class NevergradSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("nevergrad")


@DeveloperAPI
@register_search_algorithm("optuna")
@ludwig_dataclass
class OptunaSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("optuna")


@DeveloperAPI
@register_search_algorithm("skopt")
class SkoptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("skopt")


@DeveloperAPI
@register_search_algorithm("zoopt")
@ludwig_dataclass
class ZooptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("zoopt")
