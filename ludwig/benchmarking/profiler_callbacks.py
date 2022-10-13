from typing import Any, Dict

from ludwig.api_annotations import DeveloperAPI
from ludwig.benchmarking.profiler import LudwigProfiler
from ludwig.callbacks import Callback
from ludwig.constants import EVALUATION, PREPROCESSING, TRAINING


# TODO: Change annotation to PublicAPI once Ludwig 0.7 is released
@DeveloperAPI
class LudwigProfilerCallback(Callback):
    """Class that defines the methods necessary to hook into process."""

    def __init__(self, experiment: Dict[str, Any]):
        self.experiment_name = experiment["experiment_name"]
        self.use_torch_profiler = experiment["profiler"]["use_torch_profiler"]
        self.logging_interval = experiment["profiler"]["logging_interval"]
        self.preprocess_profiler = None
        self.train_profiler = None
        self.evaluation_profiler = None

    def on_preprocess_start(self, *args, **kwargs):
        self.preprocess_profiler = LudwigProfiler(
            tag=PREPROCESSING,
            output_dir=self.experiment_name,
            use_torch_profiler=self.use_torch_profiler,
            logging_interval=self.logging_interval,
        )
        self.preprocess_profiler.__enter__()

    def on_preprocess_end(self, *args, **kwargs):
        self.preprocess_profiler.__exit__(None, None, None)
        del self.preprocess_profiler

    def on_train_start(self, *args, **kwargs):
        self.train_profiler = LudwigProfiler(
            tag=TRAINING,
            output_dir=self.experiment_name,
            use_torch_profiler=self.use_torch_profiler,
            logging_interval=self.logging_interval,
        )
        self.train_profiler.__enter__()

    def on_train_end(self, *args, **kwargs):
        self.train_profiler.__exit__(None, None, None)
        del self.train_profiler

    def on_evaluation_start(self):
        self.evaluation_profiler = LudwigProfiler(
            tag=EVALUATION,
            output_dir=self.experiment_name,
            use_torch_profiler=self.use_torch_profiler,
            logging_interval=self.logging_interval,
        )
        self.evaluation_profiler.__enter__()

    def on_evaluation_end(self):
        self.evaluation_profiler.__exit__(None, None, None)
        del self.evaluation_profiler
