# !/usr/bin/env python
# Copyright (c) 2021 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from abc import ABC
from collections.abc import Callable
from typing import Any

from ludwig.api_annotations import PublicAPI
from ludwig.types import HyperoptConfigDict, ModelConfigDict, TrainingSetMetadataDict


@PublicAPI
class Callback(ABC):
    """Base class for Ludwig lifecycle callbacks.

    Override any hook methods to execute custom logic at specific points in the
    Ludwig pipeline (training, evaluation, hyperopt, etc.). All hook methods
    are no-ops by default.

    Example::

        class MyCallback(Callback):
            def on_train_start(self, model, config, config_fp, **kwargs):
                print(f"Training started with config: {config_fp}")

            def on_epoch_end(self, trainer, progress_tracker, save_path, **kwargs):
                epoch = progress_tracker.epoch
                loss = progress_tracker.train_metrics.get("combined", {}).get("loss", [])
                if loss:
                    print(f"Epoch {epoch}: loss = {loss[-1]:.4f}")

            def on_preprocess_progress(self, progress: float, **kwargs):
                print(f"Preprocessing: {progress:.0%}")

        model = LudwigModel(config, callbacks=[MyCallback()])

    The ``on_preprocess_progress`` hook fires periodically while Ludwig preprocesses
    features (during both training and prediction).  ``progress`` is a ``float``
    in ``[0.0, 1.0]`` tracking completed partitions -- one increment per pandas
    column, Dask partition, or Ray worker task.  The final call is always
    ``progress=1.0``.  Works with all backends (pandas, Ray/Dask) with no extra
    configuration required.
    """

    def on_cmdline(self, cmd: str, *args: list[str]):
        """Called when Ludwig is run from the command line.

        Args:
            cmd: The Ludwig subcommand being run (e.g., ``"train"``, ``"predict"``).
            *args: The full list of command-line arguments (``sys.argv``).
        """

    def on_preprocess_start(self, config: ModelConfigDict, **kwargs):
        """Called before preprocessing starts.

        Args:
            config: The full Ludwig config dict.
        """

    def on_preprocess_end(
        self,
        training_set,
        validation_set,
        test_set,
        training_set_metadata: TrainingSetMetadataDict,
        **kwargs,
    ):
        """Called after preprocessing ends.

        Args:
            training_set: Preprocessed training dataset.
            validation_set: Preprocessed validation dataset.
            test_set: Preprocessed test dataset.
            training_set_metadata: Metadata inferred from the training set,
                including vocabularies, feature statistics, and preprocessing
                settings (same content as ``training_set_metadata.json``).
        """

    def on_hyperopt_init(self, experiment_name: str, **kwargs):
        """Called to initialize state before hyperparameter optimization begins.

        Args:
            experiment_name: Name of the current experiment.
        """

    def on_hyperopt_preprocessing_start(self, experiment_name: str, **kwargs):
        """Called before data preprocessing for hyperparameter optimization begins.

        Args:
            experiment_name: Name of the current experiment.
        """

    def on_hyperopt_preprocessing_end(self, experiment_name: str, **kwargs):
        """Called after data preprocessing for hyperparameter optimization completes.

        Args:
            experiment_name: Name of the current experiment.
        """

    def on_hyperopt_start(self, experiment_name: str, **kwargs):
        """Called before any hyperparameter optimization trials are started.

        Args:
            experiment_name: Name of the current experiment.
        """

    def on_hyperopt_end(self, experiment_name: str, **kwargs):
        """Called after all hyperparameter optimization trials are completed.

        Args:
            experiment_name: Name of the current experiment.
        """

    def on_hyperopt_finish(self, experiment_name: str, **kwargs):
        """Deprecated — use ``on_hyperopt_end`` instead."""

    def on_hyperopt_trial_start(self, parameters: HyperoptConfigDict, **kwargs):
        """Called before the start of each hyperparameter optimization trial.

        Args:
            parameters: The full parameter dict for this hyperopt trial.
        """

    def on_hyperopt_trial_end(self, parameters: HyperoptConfigDict, **kwargs):
        """Called after the end of each hyperparameter optimization trial.

        Args:
            parameters: The full parameter dict for this hyperopt trial.
        """

    def should_stop_hyperopt(self):
        """Return ``True`` to stop the entire hyperopt run early.

        See `Ray Tune Stoppers <https://docs.ray.io/en/latest/tune/api_docs/stoppers.html>`_.
        """
        return False

    def on_resume_training(self, is_coordinator: bool, **kwargs):
        """Called when training is resumed from a checkpoint.

        Args:
            is_coordinator: Whether this worker is the coordinator.
        """

    def on_train_init(
        self,
        base_config: ModelConfigDict,
        experiment_directory: str,
        experiment_name: str,
        model_name: str,
        output_directory: str,
        resume_directory: str | None,
        **kwargs,
    ):
        """Called after preprocessing but before model and trainer objects are created.

        Args:
            base_config: User-specified config before defaults/inferred values are added.
            experiment_directory: Experiment directory (same as ``output_directory``
                when no experiment name is specified).
            experiment_name: The experiment name.
            model_name: The model name.
            output_directory: Path where training results are stored.
            resume_directory: Checkpoint directory to resume from, or ``None``.
        """

    def on_train_start(
        self,
        model,
        config: ModelConfigDict,
        config_fp: str | None,
        **kwargs,
    ):
        """Called after the trainer is created, before training begins.

        Args:
            model: The Ludwig model (``LudwigModule`` instance).
            config: The full config dict.
            config_fp: Path to the YAML config file, or ``None`` if config was
                passed as a dict.
        """

    def on_train_end(self, output_directory: str, **kwargs):
        """Called at the end of training, before the model is saved.

        Args:
            output_directory: Path where training results are stored.
        """

    def on_trainer_train_setup(self, trainer, save_path: str, is_coordinator: bool, **kwargs):
        """Called in every trainer (coordinator or worker) before training starts.

        Args:
            trainer: The trainer instance.
            save_path: Path to the directory where the model is saved.
            is_coordinator: Whether this trainer is the coordinator.
        """

    def on_trainer_train_teardown(self, trainer, progress_tracker, save_path: str, is_coordinator: bool, **kwargs):
        """Called in every trainer (coordinator or worker) after training completes.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress (epochs, metrics, etc.).
            save_path: Path to the directory where the model is saved.
            is_coordinator: Whether this trainer is the coordinator.
        """

    def on_batch_start(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator before each training batch.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_batch_end(self, trainer, progress_tracker, save_path: str, sync_step: bool = True, **kwargs):
        """Called on the coordinator after each training batch.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
            sync_step: Whether model params were updated and synced in this step.
        """

    def on_eval_start(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator at the start of evaluation.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_eval_end(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator at the end of evaluation.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_epoch_start(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator before the start of each epoch.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_epoch_end(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator after the end of each epoch.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_validation_start(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator before validation starts.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_validation_end(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator after validation completes.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_test_start(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator before test evaluation starts.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def on_test_end(self, trainer, progress_tracker, save_path: str, **kwargs):
        """Called on the coordinator after test evaluation ends.

        Args:
            trainer: The trainer instance.
            progress_tracker: Object tracking training progress.
            save_path: Path to the model save directory.
        """

    def should_early_stop(self, trainer, progress_tracker, is_coordinator, **kwargs):
        """Return ``True`` to trigger early stopping on any worker.

        Ludwig ORs the return value across all workers, so any worker returning
        ``True`` will stop training.
        """
        return False

    def on_checkpoint(self, trainer, progress_tracker, **kwargs):
        """Called after each checkpoint, regardless of whether the model was evaluated or saved."""

    def on_save_best_checkpoint(self, trainer, progress_tracker, save_path, **kwargs):
        """Called on every worker immediately after a new best model checkpoint is saved."""

    def on_build_metadata_start(self, df, mode: str, **kwargs):
        """Called before building dataset metadata.

        Args:
            df: The dataset (``pd.DataFrame``).
            mode: One of ``"prediction"``, ``"training"``, or ``None``.
        """

    def on_build_metadata_end(self, df, mode, **kwargs):
        """Called after dataset metadata has been built.

        Args:
            df: The dataset (``pd.DataFrame``).
            mode: One of ``"prediction"``, ``"training"``, or ``None``.
        """

    def on_build_data_start(self, df, mode, **kwargs):
        """Called before ``build_data`` (preprocessing, missing-value handling, metadata update).

        Args:
            df: The dataset (``pd.DataFrame``).
            mode: One of ``"prediction"``, ``"training"``, or ``None``.
        """

    def on_preprocess_progress(self, progress: float, **kwargs):
        """Called periodically during ``build_data`` to report preprocessing progress.

        Progress is tracked at the partition level: each engine partition (pandas
        column, Dask partition, or Ray task) increments the counter after it
        completes, so the value reflects actual work done rather than an estimate.

        Args:
            progress: Fraction of preprocessing completed, in the range ``[0.0, 1.0]``.
        """

    def on_build_data_end(self, df, mode, **kwargs):
        """Called after ``build_data`` completes.

        Args:
            df: The dataset (``pd.DataFrame``).
            mode: One of ``"prediction"``, ``"training"``, or ``None``.
        """

    def on_evaluation_start(self, **kwargs):
        """Called before preprocessing for standalone evaluation."""

    def on_evaluation_end(self, **kwargs):
        """Called after standalone evaluation completes."""

    def on_visualize_figure(self, fig, **kwargs):
        """Called after a visualization figure is generated.

        Args:
            fig: The generated ``matplotlib.figure.Figure``.
        """

    def on_ludwig_end(self, **kwargs):
        """Called at the very end of a Ludwig run for any cleanup."""

    def prepare_ray_tune(
        self,
        train_fn: Callable,
        tune_config: dict[str, Any],
        tune_callbacks: list[Callable],
        **kwargs,
    ) -> tuple[Callable, dict[str, Any]]:
        """Configure the Ray Tune training function and config before a hyperopt run.

        Args:
            train_fn: The function that runs a single hyperopt trial.
            tune_config: The Ray Tune configuration dict.
            tune_callbacks: Additional Ray Tune callbacks (not yet used by Ludwig).

        Returns:
            A tuple ``(train_fn, tune_config)`` — possibly modified — that is
            passed directly to Ray Tune.
        """
        return train_fn, tune_config
