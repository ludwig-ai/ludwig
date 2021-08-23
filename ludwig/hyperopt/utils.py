import json
import logging
import os


from ludwig.hyperopt.results import HyperoptResults
from ludwig.utils.data_utils import save_json
from ludwig.utils.print_utils import print_boxed

logger = logging.getLogger(__name__)


ray_hparamspace_type_map = {
    "uniform": "float",
    "quniform": "float",
    "loguniform": "float",
    "qloguniform": "float",
    "randn": "float",
    "qrandn": "float",
    "qlograndint": "int",
    "lograndint": "int",
    "qrandint": "int",
    "randint": "int",
    "grid": "category",
    "choice": "category",
    "grid_search" : "category",
}


def print_hyperopt_results(hyperopt_results: HyperoptResults):
    print_boxed('HYPEROPT RESULTS', print_fun=logger.info)
    for trial_results in hyperopt_results.ordered_trials:
        logger.info('score: {:.6f} | parameters: {}'.format(
            trial_results.metric_score, trial_results.parameters
        ))
    logger.info("")


def save_hyperopt_stats(hyperopt_stats, hyperopt_dir_name):
    hyperopt_stats_fn = os.path.join(
        hyperopt_dir_name,
        'hyperopt_statistics.json'
    )
    save_json(hyperopt_stats_fn, hyperopt_stats)


def load_json_value(v):
    try:
        return json.loads(v)
    except:
        return v


def load_json_values(d):
    return {
        k: load_json_value(v)
        for k, v in d.items()
    }
