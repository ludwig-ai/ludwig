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


## Contributors, add functions here:

class Comet():

    @staticmethod
    def import_call(argv, *args, **kwargs):
        """
        Enable Third-party support from comet.ml
        Allows experiment tracking, visualization, and
        management.
        """
        try:
            import comet_ml
            comet_ml.Experiment()
        except:
            logging.error("Ignored --comet " +
                          "See: https://www.comet.ml/" +
                          "docs/python-sdk/getting-started/ " +
                          "for more information")


## Contributors, classes and call functions here:


## Contributors, add your class here:
contrib_registry = {
    'comet': Comet
}


## NOTE: Other contribs may need more sophisticated
##       arg parsing.
def contrib_import():
    argv_list = sys.argv
    argv_set = set(argv_list)

    for contrib_name, contrib_class in contrib_registry.items():
        ## First, check for your flag(s):
        parameter_name = '--' + contrib_name
        if parameter_name in argv_set:
            ## Second, call your function:
            contrib_class.import_call(argv_list)
            ## Third, clean up and remove your flag(s)
            sys.argv.remove(parameter_name)
