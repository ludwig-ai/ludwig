import logging
import os

from ludwig.callbacks import Callback
from ludwig.utils.data_utils import chunk_dict, flatten_dict, to_json_dict
from ludwig.utils.package_utils import LazyLoader

mlflow = LazyLoader('mlflow', globals(), 'mlflow')

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name=experiment_name)


class MlflowCallback(Callback):
    def __init__(self):
        self.experiment_id = None
        self.run = None

    def on_hyperopt_init(self, experiment_name):
        self.experiment_id = _get_or_create_experiment_id(experiment_name)

    def on_train_init(
            self,
            base_config,
            experiment_name,
            output_directory,
            **kwargs
    ):
        if self.experiment_id is not None:
            # Experiment may already have been set during hyperopt init, in
            # which case we don't want to create a new experiment / run, as
            # this should be handled by the executor.
            return

        self.experiment_id = _get_or_create_experiment_id(experiment_name)
        run_name = os.path.basename(output_directory)
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        mlflow.log_dict(to_json_dict(base_config), 'config.yaml')

    def on_train_start(self, config, **kwargs):
        train_config = {'training': config['training']}
        config_flattened = flatten_dict(train_config)
        for chunk in chunk_dict(config_flattened, chunk_size=100):
            mlflow.log_params(chunk)

    def on_train_end(self, output_directory):
        for fname in os.listdir(output_directory):
            mlflow.log_artifact(os.path.join(output_directory, fname))
        if self.run is not None:
            mlflow.end_run()

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        mlflow.log_metrics(
            progress_tracker.log_metrics,
            step=progress_tracker.steps
        )

    def on_visualize_figure(self, fig):
        # TODO: need to also include a filename for this figure
        # mlflow.log_figure(fig)
        pass

    def prepare_ray_tune(self, train_fn, tune_config):
        from ray.tune.integration.mlflow import mlflow_mixin
        return mlflow_mixin(train_fn), {
            **tune_config,
            'mlflow': {
                'experiment_id': self.experiment_id,
                'tracking_uri': mlflow.get_tracking_uri(),
            }
        }
