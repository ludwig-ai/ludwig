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

class Comet():
    """
    Class that defines the methods necessary
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
            logging.error("Ignored --comet: Please install comet_ml")
            return None

        return Comet()

    def experiment(self, *args, **kwargs):
        import comet_ml
        self.experiment = comet_ml.Experiment(log_code=False)
        config = comet_ml.get_config()
        self._save_config(config)

    def train(self, *args, **kwargs):
        import comet_ml
        self.experiment = comet_ml.Experiment(log_code=False)
        config = comet_ml.get_config()
        self._save_config(config)

    def visualize(self, *args, **kwargs):
        import comet_ml
        self.experiment = comet_ml.ExistingExperiment()

    def predict(self, *args, **kwargs):
        import comet_ml
        self.experiment = comet_ml.ExistingExperiment()

    def _save_config(self, config):
        config["comet.experiment_key"] = self.experiment.id
        config.save()
