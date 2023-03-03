from dataclasses import field
from typing import Dict, List, Optional

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()


def register_search_algorithm(name: str):
    def wrap(cls):
        search_algorithm_registry[name] = cls
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
            "Name of the metric used as objective in this experiment. This metric must be present in raw_data "
            "argument to log_data. This metric must also be present in the dict reported/returned by the Trainable. "
            "If None but a mode was passed, the `ray.tune.result.DEFAULT_METRIC` will be used per default."
        ),
    )

    mode: Optional[str] = None

    points_to_evaluate: Optional[List[Dict]] = None

    parameter_constraints: Optional[List] = None

    outcome_constraints: Optional[List] = None

    ax_client = None


@DeveloperAPI
@register_search_algorithm("bayesopt")
@ludwig_dataclass
class BayesOptSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bayesopt")


@DeveloperAPI
@register_search_algorithm("blendsearch")
@ludwig_dataclass
class BlendsearchSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("blendsearch")


@DeveloperAPI
@register_search_algorithm("bohb")
@ludwig_dataclass
class BOHBSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("bohb")


@DeveloperAPI
@register_search_algorithm("cfo")
@ludwig_dataclass
class CFOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("cfo")


@DeveloperAPI
@register_search_algorithm("dragonfly")
@ludwig_dataclass
class DragonflySAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("dragonfly")


@DeveloperAPI
@register_search_algorithm("hebo")
@ludwig_dataclass
class HEBOSAConfig(BaseSearchAlgorithmConfig):
    type: str = schema_utils.ProtectedString("hebo")


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
