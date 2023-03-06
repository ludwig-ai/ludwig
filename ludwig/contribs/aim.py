import json
import logging

from ludwig.api_annotations import PublicAPI
from ludwig.callbacks import Callback
from ludwig.utils.data_utils import NumpyEncoder
from ludwig.utils.package_utils import LazyLoader

aim = LazyLoader("aim", globals(), "aim")

logger = logging.getLogger(__name__)


@PublicAPI
class AimCallback(Callback):
    """Class that defines the methods necessary to hook into process."""

    def __init__(self, repo=None):
        self.repo = repo

    def on_train_init(
        self,
        base_config,
        experiment_directory,
        experiment_name,
        model_name,
        output_directory,
        resume_directory,
    ):
        logger.info("aim.on_train_init() called...")

        try:
            query = f'run.name == "{model_name}"'
            if self.repo is None:
                aim_repo = aim.Repo.default_repo()
            else:
                aim_repo = aim.Repo.from_path(self.repo)
            runs_generator = aim_repo.query_runs(query)
            run = next(runs_generator.iter_runs())
            run_hash = run.run.hash
            self.aim_run = aim.Run(run_hash=run_hash, repo=self.repo, experiment=experiment_name)
        except Exception:
            self.aim_run = aim.Run(repo=self.repo, experiment=experiment_name)
            self.aim_run.name = model_name

        self.aim_run["base_config"] = self.normalize_config(base_config)

        params = dict(name=model_name, dir=experiment_directory)
        self.aim_run["params"] = params

    def aim_track(self, progress_tracker):
        logger.info(f"aim.aim_track() called for epoch {progress_tracker.epoch}, step: {progress_tracker.steps}")

        if self.aim_run:
            for key, value in progress_tracker.log_metrics().items():
                if "metrics" in key and "best" not in key:
                    metrics_dict_name, feature_name, metric_name = key.split(".")

                    self.aim_run.track(
                        value,
                        name=metric_name,
                        context={metrics_dict_name: feature_name},
                        epoch=progress_tracker.epoch,
                        step=progress_tracker.steps,
                    )

    def on_trainer_train_teardown(self, trainer, progress_tracker, save_path, is_coordinator: bool):
        pass

    def on_train_start(self, model, config, *args, **kwargs):
        logger.info("aim.on_train_start() called...")

        config = config.copy()
        del config["input_features"]
        del config["output_features"]

        self.aim_run["train_config"] = self.normalize_config(config)

    def on_train_end(self, output_directory, *args, **kwargs):
        pass

    def on_eval_end(self, trainer, progress_tracker, save_path):
        optimizer_config = {}
        for index, group in enumerate(trainer.optimizer.param_groups):
            for key in group:
                if "param" not in key:
                    optimizer_config[f"param_group_{index}_{key}"] = group[key]

        self.aim_run["optimizer_config"] = self.normalize_config(optimizer_config)

        self.aim_track(progress_tracker)

    def on_ludwig_end(self):
        self.aim_run.close()
        self.aim_run = None

    def on_visualize_figure(self, fig):
        logger.info("aim.on_visualize_figure() called...")
        if self.aim_run:
            self.aim_run.track(aim.Figure(fig), name="Figure", context={"type": "Training Figure"})

    @staticmethod
    def normalize_config(config):
        """Convert to json string and back again to remove numpy types."""
        return json.loads(json.dumps(config, cls=NumpyEncoder))
