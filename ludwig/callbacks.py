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
from typing import Any, Dict

from ludwig.data.dataset.base import Dataset


class Callback(ABC):
    def on_cmdline(self, cmd, *args):
        pass

    def on_preprocess_start(self, config: Dict[str, Any]):
        pass

    def on_preprocess_end(
        self, training_set: Dataset, validation_set: Dataset, test_set: Dataset, training_set_metadata: Dict[str, Any]
    ):
        pass

    def on_hyperopt_init(self, experiment_name):
        pass

    def on_hyperopt_preprocessing_start(self, experiment_name):
        pass

    def on_hyperopt_preprocessing_end(self, experiment_name):
        pass

    def on_hyperopt_start(self, experiment_name):
        pass

    def on_hyperopt_end(self, experiment_name):
        pass

    def on_hyperopt_finish(self, experiment_name):
        # TODO(travis): remove in favor of on_hyperopt_end for naming consistency
        pass

    def on_hyperopt_trial_start(self, parameters):
        pass

    def on_hyperopt_trial_end(self, parameters):
        pass

    def on_train_init(
        self,
        base_config,
        experiment_directory,
        experiment_name,
        model_name,
        output_directory,
        resume,
    ):
        pass

    def on_train_start(
        self,
        model,
        config,
        config_fp,
    ):
        pass

    def on_train_end(self, output_directory):
        pass

    def on_trainer_train_setup(self, trainer, save_path, is_coordinator):
        """Called in EVERY trainer (rank) before training starts."""
        pass

    def on_trainer_train_teardown(self, trainer, progress_tracker, is_coordinator):
        """Called in EVERY trainer (rank) after training completes."""
        pass

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

    def should_early_stop(self, trainer, progress_tracker, is_coordinator):
        # Triggers early stopping if any callback on any worker returns True
        return False

    def on_build_metadata_start(self, df, mode):
        pass

    def on_build_metadata_end(self, df, mode):
        pass

    def on_build_data_start(self, df, mode):
        pass

    def on_build_data_end(self, df, mode):
        pass

    def on_evaluation_start(self):
        pass

    def on_evaluation_end(self):
        pass

    def on_visualize_figure(self, fig):
        pass

    def on_ludwig_end(self):
        """Convenience method for any cleanup."""
        pass

    def prepare_ray_tune(self, train_fn, tune_config, tune_callbacks):
        """Configures Ray Tune to properly use this callback in each trial."""
        return train_fn, tune_config

    @staticmethod
    def preload():
        pass
