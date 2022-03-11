import json
import logging
import os

from ludwig.constants import HYPEROPT, PARAMETERS, PREPROCESSING
from ludwig.hyperopt.results import HyperoptResults
from ludwig.utils.data_utils import save_json
from ludwig.utils.print_utils import print_boxed

logger = logging.getLogger(__name__)


def print_hyperopt_results(hyperopt_results: HyperoptResults):
    print_boxed("HYPEROPT RESULTS", print_fun=logger.info)
    for trial_results in hyperopt_results.ordered_trials:
        logger.info(f"score: {trial_results.metric_score.value:.6f} | parameters: {trial_results.parameters}")
    logger.info("")


def save_hyperopt_stats(hyperopt_stats, hyperopt_dir_name):
    hyperopt_stats_fn = os.path.join(hyperopt_dir_name, "hyperopt_statistics.json")
    save_json(hyperopt_stats_fn, hyperopt_stats)


def load_json_value(v):
    try:
        return json.loads(v)
    except Exception as e:
        logger.warning(f"While loading json, encountered exception: {e}")
        return v


def load_json_values(d):
    return {k: load_json_value(v) for k, v in d.items()}


def should_tune_preprocessing(config):
    parameters = config[HYPEROPT][PARAMETERS]
    for param_name in parameters.keys():
        if f"{PREPROCESSING}." in param_name:
            return True
    return False
