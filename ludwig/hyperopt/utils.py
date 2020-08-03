from ludwig.constants import STRATEGY, EXECUTOR, VALIDATION, COMBINED, LOSS, \
    MINIMIZE
from ludwig.hyperopt.execution import executor_registry
from ludwig.hyperopt.sampling import sampler_registry
from ludwig.utils.misc_utils import set_default_value, set_default_values, \
    get_from_registry, get_class_attributes


def update_hyperopt_params_with_defaults(hyperopt_params):
    set_default_value(hyperopt_params, STRATEGY, {})
    set_default_value(hyperopt_params, EXECUTOR, {})
    set_default_value(hyperopt_params, "split", VALIDATION)
    set_default_value(hyperopt_params, "output_feature", COMBINED)
    set_default_value(hyperopt_params, "metric", LOSS)
    set_default_value(hyperopt_params, "goal", MINIMIZE)

    set_default_values(hyperopt_params[STRATEGY], {"type": "random"})

    strategy = get_from_registry(hyperopt_params[STRATEGY]["type"],
                                 sampler_registry)
    strategy_defaults = {k: v for k, v in strategy.__dict__.items() if
                         k in get_class_attributes(strategy)}
    set_default_values(
        hyperopt_params[STRATEGY], strategy_defaults,
    )

    set_default_values(hyperopt_params[EXECUTOR], {"type": "serial"})

    executor = get_from_registry(hyperopt_params[EXECUTOR]["type"],
                                 executor_registry)
    executor_defaults = {k: v for k, v in executor.__dict__.items() if
                         k in get_class_attributes(executor)}
    set_default_values(
        hyperopt_params[EXECUTOR], executor_defaults,
    )
