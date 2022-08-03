from abc import ABC
from typing import Any

from marshmallow_dataclass import dataclass

from ludwig.constants import LOSS
from ludwig.features.feature_registries import output_type_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.executor import ExecutorConfig, ExecutorDataclassField
from ludwig.schema.hyperopt.search_algorithm import SearchAlgorithmConfig, SearchAlgorithmDataclassField

# hyperopt:
#   search_alg:
#     type: hyperopt
#     random_state_seed: 42
#   executor:
#     type: ray
#     num_samples: 10
#     time_budget_s: 3600
#     scheduler:
#       type: async_hyperband
#       time_attr: time_total_s
#       max_t: 3600
#       grace_period: 72
#       reduction_factor: 5
#     cpu_resources_per_trial: 1
#   parameters:
#     trainer.learning_rate:
#       space: choice
#       categories:
#         - 0.005
#         - 0.01
#         - 0.02
#         - 0.025
#   output_feature: combined
#   goal: minimize
#   metric: loss


def get_hyperopt_metric_options():
    all_metrics = []
    for oftype in output_type_registry:
        ofcls = output_type_registry[oftype]
        all_metrics += ofcls.metric_functions.keys()
    return all_metrics


@dataclass
class HyperoptConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Basic hyperopt settings."""

    output_feature: str = "combined"  # TODO: make more restrictive

    goal: str = schema_utils.StringOptions(options=["minimize", "maximize"], default="minimize", allow_none=False)

    metric: str = schema_utils.StringOptions(options=get_hyperopt_metric_options(), default=LOSS, allow_none=False)

    search_alg: SearchAlgorithmConfig = SearchAlgorithmDataclassField(description="")

    executor: ExecutorConfig = ExecutorDataclassField(description="")

    parameters: Any = None


def get_hyperopt_jsonschema():
    return schema_utils.unload_jsonschema_from_marshmallow_class(HyperoptConfig)

    # def allowed_types_for_trainer_schema(cls) -> List[str]:
    #     """Returns the allowed values for the "type" field on the given trainer schema."""
    #     return cls.Schema().fields[TYPE].validate.choices

    # conds = []
    # all_trainer_types = []
    # for trainer in trainer_schema_registry:
    #     trainer_cls = trainer_schema_registry[trainer]

    #     allowed_trainer_types = allowed_types_for_trainer_schema(trainer_cls)
    #     all_trainer_types.extend(allowed_trainer_types)

    #     other_props = schema_utils.unload_jsonschema_from_marshmallow_class(trainer_cls)["properties"]
    #     other_props.pop("type")
    #     for trainer_type in allowed_trainer_types:
    #         trainer_cond = schema_utils.create_cond(
    #             {"type": trainer_type},
    #             other_props,
    #         )
    #         conds.append(trainer_cond)

    # return {
    #     "type": "object",
    #     "properties": {
    #         "type": {"type": "string", "enum": all_trainer_types},
    #     },
    #     "title": "trainer_options",
    #     "allOf": conds,
    #     "description": "Use type 'trainer' for training ECD models, or 'lightgbm_trainer' for Tree models.",
    # }
