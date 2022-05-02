import logging
import os

from ludwig.callbacks import Callback
from ludwig.utils.package_utils import LazyLoader

aim = LazyLoader("aim", globals(), "aim")

logger = logging.getLogger(__name__)


class AimCallback(Callback):
    """Class that defines the methods necessary to hook into process."""

    def on_train_init(
        self,
        base_config,
        experiment_directory,
        experiment_name,
        model_name,
        output_directory,
        resume,
    ):
        logger.info("wandb.on_train_init() called...")

        self.aim_run = aim.Run(repo=experiment_directory, experiment=experiment_name, run_hash=None)
        self.aim_run["base_config"] = base_config

        params = dict(name=model_name, dir=output_directory)
        self.aim_run["params"] = params

    def aim_track(self, progress_tracker):

        if self.aim_run:
            train_config = self.aim_run["train_config"]
            for key, value in progress_tracker.log_metrics().items():
                if "metrics" in key:
                    print(key)
                    metrics_dict_name, feature_name, metric_name = key.split(".")
                    self.aim_run.track(
                        value,
                        name=f"{metrics_dict_name}.{feature_name}",
                        context={"type": f"{key}_{metric_name}"},
                        epoch=progress_tracker.epoch,
                        step=progress_tracker.step,
                    )

                else:
                    train_config[key] = value

            self.aim_run["train_config"] = train_config

    def on_trainer_train_teardown(self, trainer, progress_tracker, save_path, is_coordinator: bool):
        optimizer_config = {}
        for group in trainer.optimizer.param_groups:
            if isinstance(group, dict):
                optimizer_config.update(group)

        self.aim_run["optimizer_config"] = optimizer_config

        self.aim_track(progress_tracker)

    def on_train_start(self, model, config, *args, **kwargs):
        logger.info("aim.on_train_start() called...")
        logger.info(args, kwargs)
        self.aim_run["train_config"] = config

    def on_ludwig_end(self):
        self.aim_run.close()
        self.aim_run = None

    def on_train_end(self, output_directory, *args, **kwargs):
        pass

    def on_eval_end(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_visualize_figure(self, fig):
        logger.info("aim.on_visualize_figure() called...")
        if self.aim_run:
            self.aim_run.track(aim.Figure(fig), name="Figure", context={"type": "Training Figure"})

    @staticmethod
    def preload():
        import aim  # noqa
