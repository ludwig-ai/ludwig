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

"""
Module for handling contributed support.
All classes must have the following functions:
- import_call: Loaded first to handle items
  that need to be setup before core modules,
  like tensorflow.
If a call doesn't apply, provide an empy
implementation with `pass`.
"""

import logging
import sys


## Contributors, add classes here:

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


## Contributors, add your class here:
contrib_registry = {
    'comet': {
        "class": Comet,
        "instance": None,
        }
}


def contrib_import():
    """
    Checks for contrib flags, and calls static method:

    ContribClass.import_call(argv_list)

    import_call() will return an instance to the class
    if appropriate (all dependencies are met, for example).
    """
    argv_list = sys.argv
    argv_set = set(argv_list)
    for contrib_name in contrib_registry:
        parameter_name = '--' + contrib_name
        if parameter_name in argv_set:
            ## Calls ContribClass.import_call(argv_list)
            ## and return an instance, if appropriate
            contrib_class = contrib_registry[contrib_name]["class"]
            instance = contrib_class.import_call(argv_list)
            ## Save instance in registry
            if instance:
                contrib_registry[contrib_name]["instance"] = instance
            ## Clean up and remove your flag
            sys.argv.remove(parameter_name)

def contrib_command(command, *args, **kwargs):
    """
    If a contrib has an instance in the registry,
    this this will call:

    ContribInstance.COMMAND(*args, **kwargs)
    """
    for contrib_name in contrib_registry:
        instance = contrib_registry[contrib_name]["instance"]
        if instance:
            method = getattr(instance, command, None)
            if method:
                method(*args, **kwargs)
