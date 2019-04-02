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
Module for handling contributed support. Loaded first
to handle items that need to be setup before core
modules, like tensorflow.
"""

import sys
import logging

## Contributors, add functions here:

def enable_comet_support():
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


## Contributors, call functions here:

## NOTE: Other contribs may need more sophisticated
##       arg parsing.
for arg in list(sys.argv):
    ## First, check for your flag(s):
    if arg == "--comet":
        ## Second, call your function:
        enable_comet_support()
        ## Third, clean up and remove your flag(s)
        sys.argv.remove("--comet")
    ## elif ... other flags
