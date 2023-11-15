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
from datetime import datetime
import json
import requests

from ludwig.api_annotations import PublicAPI
from ludwig.callbacks import Callback
from ludwig.utils.package_utils import LazyLoader

logger = logging.getLogger(__name__)


@PublicAPI
class PredibaseCallback(Callback):
    """Class that defines the methods necessary to hook into process."""

    def __init__(self, base_uri: str = "http://localhost:8082"):
        self.base_uri=base_uri
        pass
        
    def on_train_init(
        self,
        base_config,
        experiment_directory,
        experiment_name,
        model_name,
        output_directory,
        resume_directory,
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        print("train init")

    def on_train_start(self, model, config, config_fp, *args, **kwargs):
        res = self.publish_message({ "event": "train_start" })
        print("train start", res)

    def on_train_end(self, output_directory, *args, **kwargs):
        res = self.publish_message({ "event": "train_end" })
        print("train end", res)

    def on_eval_end(self, trainer, progress_tracker, save_path):
        metrics = self.get_metrics(progress_tracker)
        res = self.publish_message({ "event": "eval_end", "metrics": metrics })
        print("eval end", res)

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        metrics = self.get_metrics(progress_tracker)
        res = self.publish_message({ "event": "epoch_end", "metrics": metrics })
        print("epoch end", res)

    def on_visualize_figure(self, fig):
        pass

    def on_cmdline(self, cmd, *args):
        pass

    def get_metrics(self, progress_tracker): 
        return dict([(key, value) for key, value in progress_tracker.log_metrics().items()])

    def publish_message(self, payload: dict):
        """Publish message to redpanda/kafka queue"""
        topic = f"{self.experiment_name}-{self.model_name}"
        return requests.post(
            url=f"{self.base_uri}/topics/{topic}",
            data=json.dumps(
                dict(records=[
                    dict(value=payload, partition=0)
                ])),
            headers={"Content-Type": "application/vnd.kafka.json.v2+json"}).json()
    