import logging

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
        resume_directory,
    ):
        logger.info("aim.on_train_init() called...")

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
                        name=metric_name,
                        context={metrics_dict_name: feature_name},
                        epoch=progress_tracker.epoch,
                        step=progress_tracker.steps,
                    )
                else:
                    train_config[key] = value

            self.aim_run["train_config"] = train_config

    def on_trainer_train_teardown(self, trainer, progress_tracker, save_path, is_coordinator: bool):
        pass

    def on_train_start(self, model, config, *args, **kwargs):
        logger.info("aim.on_train_start() called...")

        config = config.copy()
        del config["input_features"]
        del config["output_features"]

        print(config)
        self.aim_run["train_config"] = config

    def on_train_end(self, output_directory, *args, **kwargs):
        pass

    def on_eval_end(self, trainer, progress_tracker, save_path):
        optimizer_config = {}
        for index, group in enumerate(trainer.optimizer.param_groups):
            for key in group:
                if "param" not in key:
                    optimizer_config[f"param_group_{index}_{key}"] = group[key]

        self.aim_run["optimizer_config"] = optimizer_config

        self.aim_track(progress_tracker)

    def on_ludwig_end(self):
        self.aim_run.close()
        self.aim_run = None

    def on_visualize_figure(self, fig):
        logger.info("aim.on_visualize_figure() called...")
        if self.aim_run:
            self.aim_run.track(aim.Figure(fig), name="Figure", context={"type": "Training Figure"})

    @staticmethod
    def preload():
        import aim  # noqa
