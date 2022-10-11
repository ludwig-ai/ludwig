from typing import Any, Dict, Union

from ludwig.backend import Backend
from ludwig.constants import CPU_RESOURCES_PER_TRIAL, EXECUTOR


def get_max_concurrent_trials(backend: Backend, hyperopt_config: Dict[str, Any]) -> Union[int, None]:
    """Returns the maximum number of concurrent trials that can be run on the backend with the available
    resources."""
    cpus_per_trial = hyperopt_config[EXECUTOR].get(CPU_RESOURCES_PER_TRIAL, 1)
    num_cpus_available = backend.get_available_resources().cpus

    # No actors will compete for ray datasets tasks dataset tasks are cpu bound
    if cpus_per_trial == 0:
        return None

    if num_cpus_available < 2:
        raise RuntimeError("At least 2 CPUs are required for hyperopt when using a HorovodBackend.")

    # Ray requires at least 2 free CPUs to ensure trials don't stall
    max_possible_trials = (num_cpus_available - 2) // cpus_per_trial

    # Users may be using an autoscaling cluster, so return None
    if max_possible_trials < 1:
        return None

    return max_possible_trials
