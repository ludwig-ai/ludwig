import logging

from ludwig.callbacks import Callback
from ludwig.utils.exceptions import ExceptionVariable

try:
    import mlflow
except Exception as e:
    mlflow = ExceptionVariable(e)

logger = logging.getLogger(__name__)


class MlflowCallback(Callback):
    def __init__(self):
        self.experiment_id = None

    def on_train_init(self, experiment_name, **kwargs):
        if self.experiment_id:
            return
        self.experiment_id = mlflow.create_experiment(experiment_name)

    def on_batch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_batch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_start(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_end(self, trainer, progress_tracker, save_path):
        pass

    def on_test_start(self, trainer, progress_tracker, save_path):
        pass

    def on_test_end(self, trainer, progress_tracker, save_path):
        pass
