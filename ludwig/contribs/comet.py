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
from datetime import datetime

logger = logging.getLogger(__name__)


class Comet():
    """
    Class that defines the methods necessary to hook into process.
    """

    @staticmethod
    def import_call(*args, **kwargs):
        """
        Enable Third-party support from comet.ml
        Allows experiment tracking, visualization, and
        management.
        """
        try:
            import comet_ml
        except ImportError:
            logger.error(
                "Ignored --comet: Please install comet_ml; see www.comet.ml")
            return None

        try:
            version = [int(i) for i in comet_ml.__version__.split(".")]
        except Exception:
            version = None
        if version is not None and version >= [1, 0, 51]:
            return Comet()
        else:
            logger.error("Ignored --comet: Need version 1.0.51 or greater")

    def __init__(self):
        self.cometml_experiment = None

    def experiment(self, *args, **kwargs):
        import comet_ml
        try:
            self.cometml_experiment = comet_ml.Experiment(log_code=False)
        except Exception:
            self.cometml_experiment = None
            logger.exception(
                "comet_ml.Experiment() had errors. Perhaps you need to define COMET_API_KEY")
            return

        logger.info("comet.experiment() called......")
        cli = self._make_command_line(args)
        self.cometml_experiment.set_code(cli)
        self.cometml_experiment.set_filename("Ludwig CLI")
        self._log_html(cli)
        config = comet_ml.get_config()
        self._save_config(config)

    def train(self, *args, **kwargs):
        import comet_ml
        try:
            self.cometml_experiment = comet_ml.Experiment(log_code=False)
        except Exception:
            self.cometml_experiment = None
            logger.exception(
                "comet_ml.Experiment() had errors. Perhaps you need to define COMET_API_KEY")
            return

        logger.info("comet.train() called......")
        cli = self._make_command_line(args)
        self.cometml_experiment.set_code(cli)
        self.cometml_experiment.set_filename("Ludwig CLI")
        self._log_html(cli)
        config = comet_ml.get_config()
        self._save_config(config)

    def train_init(self, experiment_directory, experiment_name, model_name,
                   resume, output_directory):
        if self.cometml_experiment:
            # Comet ML already initialized
            return

        import comet_ml
        try:
            self.cometml_experiment = comet_ml.Experiment(log_code=False,
                                                          project_name=experiment_name)
        except Exception:
            self.cometml_experiment = None
            logger.exception(
                "comet_ml.Experiment() had errors. Perhaps you need to define COMET_API_KEY")
            return

        logger.info("comet.train_init() called......")
        self.cometml_experiment.set_name(model_name)
        self.cometml_experiment.set_filename("Ludwig API")
        config = comet_ml.get_config()
        self._save_config(config, directory=experiment_directory)

    def train_model(self, model, config, config_path,
                    *args, **kwargs):
        logger.info("comet.train_model() called......")
        if self.cometml_experiment:
            # todo v0.4: currently not clear way to set model graph
            # see: https://github.com/comet-ml/issue-tracking/issues/296
            # if model:
            #     self.cometml_experiment.set_model_graph(
            #         str(model._graph.as_graph_def()))

            if config:
                if config_path:
                    base_name = os.path.basename(config_path)
                else:
                    base_name = "config.yaml"
                if "." in base_name:
                    base_name = base_name.rsplit(".", 1)[0] + ".json"
                else:
                    base_name = base_name + ".json"
                self.cometml_experiment.log_asset_data(config,
                                                       base_name)

    def train_save(self, output_directory, *args, **kwargs):
        logger.info("comet.train_save() called......")
        if self.cometml_experiment:
            self.cometml_experiment.log_asset_folder(output_directory)

    def train_epoch_end(self, progress_tracker):
        """
        Called from ludwig/models/model.py
        """
        logger.info("comet.train_epoch_end() called......")
        if self.cometml_experiment:
            for item_name in ["batch_size", "epoch", "steps",
                              "last_improvement_epoch",
                              "learning_rate", "best_valid_metric",
                              "num_reductions_lr",
                              "num_increases_bs", "train_metrics",
                              "vali_metrics",
                              "test_metrics"]:
                try:
                    item = getattr(progress_tracker, item_name)
                    if isinstance(item, dict):
                        for key in item:
                            if isinstance(item[key], dict):
                                for key2 in item[key]:
                                    self.cometml_experiment.log_metric(
                                        item_name + "." + key + "." + key2,
                                        item[key][key2][-1])
                            else:
                                self.cometml_experiment.log_metric(
                                    item_name + "." + key, item[key][-1])
                    elif item is not None:
                        self.cometml_experiment.log_metric(item_name, item)
                except Exception:
                    logger.info("comet.train_epoch_end() skip logging '%s'",
                                item_name)

    def experiment_save(self, *args, **kwargs):
        logger.info("comet.experiment_save() called......")
        output_directory = args[0]
        if self.cometml_experiment:
            self.cometml_experiment.log_asset_folder(output_directory)

    def visualize(self, *args, **kwargs):
        import comet_ml
        try:
            self.cometml_experiment = comet_ml.ExistingExperiment()
        except Exception:
            self.cometml_experiment = None
            logger.error("Ignored --comet. No '.comet.config' file")
            return

        logger.info("comet.visualize() called......")
        cli = self._make_command_line(args)
        self._log_html(cli)

    def visualize_figure(self, fig):
        logger.info("comet.visualize_figure() called......")
        if self.cometml_experiment:
            self.cometml_experiment.log_figure(fig)

    def predict(self, *args, **kwargs):
        import comet_ml
        try:
            self.cometml_experiment = comet_ml.ExistingExperiment()
        except Exception:
            self.cometml_experiment = None
            logger.error("Ignored --comet. No '.comet.config' file")
            return

        logger.info("comet.predict() called......")
        cli = self._make_command_line(args)
        self._log_html(cli)

    def evaluate(self, *args, **kwargs):
        import comet_ml
        try:
            self.cometml_experiment = comet_ml.ExistingExperiment()
        except Exception:
            self.cometml_experiment = None
            logger.error("Ignored --comet. No '.comet.config' file")
            return

        logger.info("comet.evaluate() called......")
        cli = self._make_command_line(args)
        self._log_html(cli)

    def _save_config(self, config, directory='.'):
        ## save the .comet.config here:
        config["comet.experiment_key"] = self.cometml_experiment.id
        config.save(directory=directory)

    def _log_html(self, text):
        ## log the text to the html tab:
        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y %H:%M:%S")
        self.cometml_experiment.log_html(
            "<p><b>%s</b>: %s</p>" % (timestamp, text))

    def _make_command_line(self, args):
        ## put the commet flag back in:
        return " ".join(list(args[:2]) + ["--comet"] + list(args[2:]))
