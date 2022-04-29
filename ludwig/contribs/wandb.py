# Copyright (c) 2019 Uber Technologies, Inc.
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
import logging
import os

from ludwig.callbacks import Callback
from ludwig.utils.package_utils import LazyLoader

wandb = LazyLoader("wandb", globals(), "wandb")

logger = logging.getLogger(__name__)


class WandbCallback(Callback):
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
        logger.info("wandb.on_train_init() called...")
        wandb.init(
            project=os.getenv("WANDB_PROJECT", experiment_name),
            name=model_name,
            sync_tensorboard=True,
            dir=output_directory,
        )
        wandb.save(os.path.join(experiment_directory, "*"))

    def on_train_start(self, model, config, *args, **kwargs):
        logger.info("wandb.on_train_start() called...")
        config = config.copy()
        del config["input_features"]
        del config["output_features"]
        wandb.config.update(config)

    def on_visualize_figure(self, fig):
        logger.info("wandb.on_visualize_figure() called...")
        if wandb.run:
            wandb.log({"figure": fig})

    @staticmethod
    def preload():
        import wandb  # noqa
