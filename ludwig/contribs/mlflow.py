import logging

from ludwig.callbacks import Callback
from ludwig.utils.data_utils import json_normalize
from ludwig.utils.exceptions import ExceptionVariable

try:
    import mlflow
except Exception as e:
    mlflow = ExceptionVariable(e)

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name=experiment_name)


class MlflowCallback(Callback):
    def __init__(self):
        self.experiment_id = None

    def on_hyperopt_init(self, experiment_name):
        self.experiment_id = _get_or_create_experiment_id(experiment_name)

    def on_train_init(self, experiment_name, **kwargs):
        if self.experiment_id is not None:
            return
        self.experiment_id = _get_or_create_experiment_id(experiment_name)

    def on_train_start(self, config, **kwargs):
        config_flattened = json_normalize(config)
        mlflow.log_params(config_flattened)

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        mlflow.log_metrics(progress_tracker.log_metrics)

    def prepare_ray_tune(self, train_fn, tune_config):
        from ray.tune.integration.mlflow import mlflow_mixin
        return mlflow_mixin(train_fn), {
            **tune_config,
            'mlflow': {
                'experiment_id': self.experiment_id,
                'tracking_uri': mlflow.get_tracking_uri(),
            }
        }
