from dataclasses import field
from importlib import import_module
from typing import Dict, List, Optional

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt import utils as hyperopt_utils
from ludwig.schema.utils import ludwig_dataclass


def points_to_evaluate_field(description: Optional[str] = None) -> fields.Field:
    return schema_utils.DictList(
        description=description
        or (
            "Initial parameter suggestions to be run first. This is for when you already have some good parameters "
            "you want to run first to help the algorithm make better suggestions for future parameters. Needs to be "
            "a list of dicts containing the configurations."
        ),
    )


def evaluated_rewards_field(description: Optional[str] = None) -> fields.Field:
    return schema_utils.List(
        description=description
        or (
            "If you have previously evaluated the parameters passed in as points_to_evaluate you can avoid re-running "
            "those trials by passing in the reward attributes as a list so the optimiser can be told the results "
            "without needing to re-compute the trial. Must be the same length as `points_to_evaluate`."
        )
    )


@DeveloperAPI
@ludwig_dataclass
class BaseSearchAlgorithmConfig(schema_utils.BaseMarshmallowConfig):
    """Basic search algorithm settings."""

    type: str = schema_utils.String(default="hyperopt", description="The search algorithm to use.")

    def set_random_state(self, ludwig_random_state: int) -> None:
        """Overwrite the config random state.

        Search algorithms refer to random state by different names, however we want to overwrite unset random states
        with the Ludwig random state. This method uses a registry of random state field names to provide a single
        interface across all search algorithms.
        """
        rs_field = hyperopt_utils.get_search_algorithm_random_state_field(self.type)
        if rs_field is not None and self.__getattribute__(rs_field) is None:
            self.__setattr__(rs_field, ludwig_random_state)

    def dependencies_installed(self) -> bool:
        """Some search algorithms require additional packages to be installed, check that they are available."""
        missing_packages = []
        missing_installs = []
        for package_name, install_name in hyperopt_utils.get_search_algorithm_dependencies(self.type):
            try:
                import_module(package_name)
            except ImportError:
                missing_packages.append(package_name)
                missing_installs.append(install_name)

        if missing_packages:
            missing_packages = ", ".join(missing_packages)
            missing_installs = " ".join(missing_installs)
            raise ImportError(
                f"Some packages needed to use hyperopt search algorithm {self.type} are not installed: "
                f"{missing_packages}. To add these dependencies, run `pip install {missing_installs}`. For more "
                "details, please refer to Ray Tune documentation for this search algorithm."
            )
        return True


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
                # **schema_utils.unload_jsonschema_from_marshmallow_class(BaseSearchAlgorithmConfig),
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(hyperopt_utils.search_algorithm_config_registry.keys()),
                        "default": default["type"],
                        "description": "The type of scheduler to use during hyperopt",
                    },
                },
                "title": "search_algorithm_options",
                "required": ["type"],
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
@hyperopt_utils.register_search_algorithm_config("random", random_state_field="random_state")
@hyperopt_utils.register_search_algorithm_config("variant_generator", random_state_field="random_state")
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
@hyperopt_utils.register_search_algorithm_config(
    "ax", dependencies=[("ax", "ax-platform"), ("sqlalchemy", "sqlalchemy")]
)
@ludwig_dataclass
class AxSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("ax")

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            r"Parameters in the experiment search space. Required elements in the dictionaries are: \“name\” (name of "
            r"this parameter, string), \“type\” (type of the parameter: \“range\”, \“fixed\”, or \“choice\”, string), "
            r"\“bounds\” for range parameters (list of two values, lower bound first), \“values\” for choice "
            r"parameters (list of values), and \“value\” for fixed parameters (single value)."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    parameter_constraints: Optional[List] = schema_utils.List(
        description=r"Parameter constraints, such as \“x3 >= x4\” or \“x3 + x4 >= 2\”."
    )

    outcome_constraints: Optional[List] = schema_utils.List(
        description=r"Outcome constraints of form \“metric_name >= bound\”, like \“m1 <= 3.\”"
    )


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config(
    "bayesopt", random_state_field="random_state", dependencies=[("bayes_opt", "bayesian-optimization")]
)
@ludwig_dataclass
class BayesOptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bayesopt")

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "Continuous search space. Parameters will be sampled from this space which will be used to run trials"
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    utility_kwargs: Optional[Dict] = schema_utils.Dict(
        description=(
            "Parameters to define the utility function. The default value is a dictionary with three keys: "
            "- kind: ucb (Upper Confidence Bound) - kappa: 2.576 - xi: 0.0"
        )
    )

    random_state: int = schema_utils.Integer(default=None, allow_none=True, description="Used to initialize BayesOpt.")

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
@hyperopt_utils.register_search_algorithm_config("blendsearch", dependencies=[("flaml", "flaml[blendsearch]")])
@ludwig_dataclass
class BlendsearchSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("blendsearch")


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config(
    "bohb", random_state_field="seed", dependencies=[("hpbandster", "hpbandster"), ("ConfigSpace", "ConfigSpace")]
)
@ludwig_dataclass
class BOHBSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bohb")

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "Continuous ConfigSpace search space. Parameters will be sampled from this space which will be used "
            "to run trials."
        )
    )

    bohb_config: Optional[Dict] = schema_utils.Dict(description="configuration for HpBandSter BOHB algorithm")

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

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
@hyperopt_utils.register_search_algorithm_config("cfo", dependencies=[("flaml", "flaml")])
@ludwig_dataclass
class CFOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("cfo")


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config(
    "dragonfly", random_state_field="random_state_seed", dependencies=[("dragonfly", "dragonfly-opt")]
)
@ludwig_dataclass
class DragonflySAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("dragonfly")

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
            "Optional domain. Should only be set if you don't pass an optimizer as the `optimizer` argument. If set, "
            "has to be one of `[cartesian, euclidean]`."
        ),
    )

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Search space. Should only be set if you don't pass an optimizer as the `optimizer` argument. Defines the "
            "search space and requires a `domain` to be set. Can be automatically converted from the `param_space` "
            "dict passed to `tune.Tuner()`."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    evaluated_rewards: Optional[List] = evaluated_rewards_field()

    random_state_seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Seed for reproducible results. Defaults to None. Please note that setting this to a value will change "
            "global random state for `numpy` on initalization and loading from checkpoint."
        ),
    )


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config(
    "hebo", random_state_field="random_state_seed", dependencies=[("hebo", "HEBO")]
)
@ludwig_dataclass
class HEBOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("hebo")

    space: Optional[List[Dict]] = schema_utils.DictList(
        description="A dict mapping parameter names to Tune search spaces or a HEBO DesignSpace object."
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    evaluated_rewards: Optional[List] = evaluated_rewards_field()

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
@hyperopt_utils.register_search_algorithm_config(
    "hyperopt", random_state_field="random_state_seed", dependencies=[("hyperopt", "hyperopt")]
)
@ludwig_dataclass
class HyperoptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("hyperopt")

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "HyperOpt configuration. Parameters will be sampled from this configuration and will be used to override "
            "parameters generated in the variant generation process."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    n_initial_points: int = schema_utils.PositiveInteger(
        default=20,
        description=(
            "The number of random evaluations of the objective function before starting to approximate it with tree "
            "parzen estimators. Defaults to 20."
        ),
    )

    random_state_seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=("Seed for reproducible results. Defaults to None."),
    )

    gamma: float = schema_utils.FloatRange(
        min=0.0,
        max=1.0,
        default=0.25,
        description=(
            "The split to use in TPE. TPE models two splits of the evaluated hyperparameters: the top performing "
            "`gamma` percent, and the remaining examples. For more details, see [Making a Science of Model Search: "
            "Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures.]"
            "(http://proceedings.mlr.press/v28/bergstra13.pdf)."
        ),
    )


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config("nevergrad", dependencies=[("nevergrad", "nevergrad")])
@ludwig_dataclass
class NevergradSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("nevergrad")

    # TODO: Add a registry mapping string names to nevergrad optimizers
    # optimizer: Optional[str] = None

    # TODO: Add schemas for nevergrad optimizer kwargs
    optimizer_kwargs: Optional[Dict] = schema_utils.Dict(
        description="Kwargs passed in when instantiating the optimizer."
    )

    space: Optional[List[Dict]] = schema_utils.DictList(
        description=(
            "Nevergrad parametrization to be passed to optimizer on instantiation, or list of parameter names if you "
            "passed an optimizer object."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config(
    "optuna", random_state_field="seed", dependencies=[("optuna", "optuna")]
)
@ludwig_dataclass
class OptunaSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("optuna")

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "Hyperparameter search space definition for Optuna's sampler. This can be either a dict with parameter "
            "names as keys and optuna.distributions as values, or a Callable - in which case, it should be a "
            "define-by-run function using optuna.trial to obtain the hyperparameter values. The function should "
            "return either a dict of constant values with names as keys, or None. For more information, see "
            "[the Optuna docs]"
            "(https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    # TODO: Add a registry of Optuna samplers schemas
    # sampler = None

    seed: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description=(
            "Seed to initialize sampler with. This parameter is only used when `sampler=None`. In all other cases, "
            "the sampler you pass should be initialized with the seed already."
        ),
    )

    evaluated_rewards: Optional[List] = evaluated_rewards_field()


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config("skopt", dependencies=[("skopt", "scikit-optimize")])
class SkoptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("skopt")

    optimizer = None

    space: Optional[Dict] = schema_utils.Dict(
        description=(
            "A dict mapping parameter names to valid parameters, i.e. tuples for numerical parameters and lists "
            "for categorical parameters. If you passed an optimizer instance as the optimizer argument, this should "
            "be a list of parameter names instead."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    evaluated_rewards: Optional[List] = evaluated_rewards_field(
        description=(
            "If you have previously evaluated the parameters passed in as points_to_evaluate you can avoid "
            "re-running those trials by passing in the reward attributes as a list so the optimiser can be told the "
            "results without needing to re-compute the trial. Must be the same length as points_to_evaluate. (See "
            "tune/examples/skopt_example.py)"
        )
    )

    convert_to_python: bool = schema_utils.Boolean(
        default=True,
        description="SkOpt outputs numpy primitives (e.g. `np.int64`) instead of Python types. If this setting is set "
        "to `True`, the values will be converted to Python primitives.",
    )


@DeveloperAPI
@hyperopt_utils.register_search_algorithm_config("zoopt", dependencies=[("zoopt", "zoopt")])
@ludwig_dataclass
class ZooptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("zoopt")

    algo: str = schema_utils.ProtectedString(
        pstring="asracos",
        description="To specify an algorithm in zoopt you want to use. Only support ASRacos currently.",
    )

    budget: Optional[int] = schema_utils.PositiveInteger(
        default=None, allow_none=True, description="Optional. Number of samples."
    )

    dim_dict: Optional[Dict] = schema_utils.Dict(
        description=(
            "Dimension dictionary. For continuous dimensions: (continuous, search_range, precision); For discrete "
            "dimensions: (discrete, search_range, has_order); For grid dimensions: (grid, grid_list). More details "
            "can be found in zoopt package."
        )
    )

    points_to_evaluate: Optional[List[Dict]] = points_to_evaluate_field()

    parallel_num: int = schema_utils.PositiveInteger(
        default=1,
        description=(
            "How many workers to parallel. Note that initial phase may start less workers than this number. More "
            "details can be found in zoopt package."
        ),
    )
