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

from datetime import datetime

class Comet():
    """
    Class that defines the methods necessary to hook into process.
    """
    @staticmethod
    def import_call(argv, *args, **kwargs):
        """
        Enable Third-party support from comet.ml
        Allows experiment tracking, visualization, and
        management.
        """
        try:
            import comet_ml
        except ImportError:
            logging.error("Ignored --comet: Please install comet_ml; see www.comet.ml")
            return None

        return Comet()

    def experiment(self, *args, **kwargs):
        import comet_ml
        try:
            self.experiment = comet_ml.Experiment(log_code=False)
        except Exception:
            logging.error("comet_ml.Experiment() had errors. Perhaps you need to define COMET_API_KEY")
            return

        cli = self._make_command_line(args)
        self.experiment.set_code(cli)
        self.experiment.set_filename("Ludwig CLI")
        self._log_html(cli)
        config = comet_ml.get_config()
        self._save_config(config)

    def train(self, *args, **kwargs):
        import comet_ml
        try:
            self.experiment = comet_ml.Experiment(log_code=False)
        except Exception:
            logging.error("comet_ml.Experiment() had errors. Perhaps you need to define COMET_API_KEY")
            return

        cli = self._make_command_line(args)
        self.experiment.set_code(cli)
        self.experiment.set_filename("Ludwig CLI")
        self._log_html(cli)
        config = comet_ml.get_config()
        self._save_config(config)

    def visualize(self, *args, **kwargs):
        import comet_ml
        try:
            self.experiment = comet_ml.ExistingExperiment()
        except Exception:
            logging.error("Ignored --comet. No '.comet.config' file")
            return

        cli = self._make_command_line(args)
        self._log_html(cli)

    def predict(self, *args, **kwargs):
        import comet_ml
        try:
            self.experiment = comet_ml.ExistingExperiment()
        except Exception:
            logging.error("Ignored --comet. No '.comet.config' file")
            return

        cli = self._make_command_line(args)
        self._log_html(cli)

    def visualize_figure(self, fig):
        if self.experiment:
            self.experiment.log_figure(fig)

    def _save_config(self, config):
        ## save the .comet.config here:
        config["comet.experiment_key"] = self.experiment.id
        config.save()

    def _log_html(self, text):
        ## log the text to the html tab:
        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y %H:%M:%S")
        self.experiment.log_html("<p><b>%s</b>: %s</p>" % (timestamp, text))

    def _make_command_line(self, args):
        ## put the commet flag back in:
        return " ".join(list(args[:2]) + ["--comet"] + list(args[2:]))
