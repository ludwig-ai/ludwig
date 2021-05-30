import logging

from ludwig.callbacks import Callback
from ludwig.utils.data_utils import json_normalize
from ludwig.utils.exceptions import ExceptionVariable

try:
    import mlflow
except Exception as e:
    mlflow = ExceptionVariable(e)

logger = logging.getLogger(__name__)


class MlflowCallback(Callback):
    def __init__(self):
        self.experiment_id = None

    def on_hyperopt_init(self, experiment_name):
        self.experiment_id = mlflow.create_experiment(experiment_name)

    def on_train_init(self, experiment_name, **kwargs):
        if self.experiment_id is not None:
            return
        self.experiment_id = mlflow.create_experiment(experiment_name)

    def on_train_start(self, config, **kwargs):
        config_flattened = json_normalize(config)
        mlflow.log_params(config_flattened)

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        mlflow.log_metrics(progress_tracker.log_metrics)
