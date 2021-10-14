# !/usr/bin/env python
# coding=utf-8
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


class Callback(ABC):
    def on_cmdline(self, cmd, *args):
        pass

    def on_hyperopt_init(self, experiment_name):
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

    def on_trainer_train_setup(self, trainer, save_path):
        """Called in EVERY trainer (rank) before training starts."""
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

    def on_visualize_figure(self, fig):
        pass

    def prepare_ray_tune(self, train_fn, tune_config, tune_callbacks):
        """Configures Ray Tune to properly use this callback in each trial."""
        return train_fn, tune_config

    @staticmethod
    def preload():
        pass
