from dataclasses import field
from typing import Dict

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
    pass


@DeveloperAPI
@register_search_algorithm("ax")
@ludwig_dataclass
class AxSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("bayesopt")
@ludwig_dataclass
class BayesOptSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("blendsearch")
@ludwig_dataclass
class BlendsearchSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("bohb")
@ludwig_dataclass
class BOHBSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("cfo")
@ludwig_dataclass
class CFOSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("dragonfly")
@ludwig_dataclass
class DragonflySAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("hebo")
@ludwig_dataclass
class HEBOSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("hyperopt")
@ludwig_dataclass
class HyperoptSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("nevergrad")
@ludwig_dataclass
class NevergradSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("optuna")
@ludwig_dataclass
class OptunaSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("skopt")
class SkoptSAConfig(BaseSearchAlgorithmConfig):
    pass


@DeveloperAPI
@register_search_algorithm("zoopt")
@ludwig_dataclass
class ZooptSAConfig(BaseSearchAlgorithmConfig):
    pass
