# coding=utf-8
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

logger = logging.getLogger(__name__)


class Wandb():
    """Class that defines the methods necessary to hook into process."""

    @staticmethod
    def import_call(*args, **kwargs):
        """
        Enable Third-party support from wandb.ai
        Allows experiment tracking, visualization, and
        management.
        """
        try:
            import wandb
            # Needed to call an attribute of wandb to make DeepSource not complain
            return Wandb() if wandb.__version__ else None
        except ImportError:
            logger.error(
                "Ignored --wandb: Please install wandb; see https://docs.wandb.com")
            return None

    def train_model(self, model, config, *args, **kwargs):
        import wandb
        logger.info("wandb.train_model() called...")
        config = config.copy()
        del config["input_features"]
        del config["output_features"]
        wandb.config.update(config)

    def train_init(self, experiment_directory, experiment_name, model_name,
                   resume, output_directory):
        import wandb
        logger.info("wandb.train_init() called...")
        wandb.init(project=os.getenv("WANDB_PROJECT", experiment_name),
                   name=model_name, sync_tensorboard=True,
                   dir=output_directory)
        wandb.save(os.path.join(experiment_directory, "*"))

    def visualize_figure(self, fig):
        import wandb
        logger.info("wandb.visualize_figure() called...")
        if wandb.run:
            wandb.log({"figure": fig})
