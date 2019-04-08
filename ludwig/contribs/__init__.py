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
All classes must have the following functions:
- import_call: Loaded first to handle items
  that need to be setup before core modules,
  like tensorflow.
If a call doesn't apply, provide an empy
implementation with `pass`.
"""

## Contributors, import your class here:
from .comet import Comet

contrib_registry = {
    ## Contributors, add your class here:
    'classes': {
        'comet': Comet,
    },
    'instances': [],
}
