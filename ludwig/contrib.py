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

"""Module for handling contributed support."""

from .contribs import contrib_registry


def add_contrib_callback_args(parser):
    for contrib_name, contrib_cls in contrib_registry.items():
        parser.add_argument(
            f"--{contrib_name}",
            dest="callbacks",
            action="append_const",
            const=contrib_cls(),
        )


def preload(argv):
    for arg in argv:
        if arg.startswith("--"):
            arg = arg[2:]

        if arg in contrib_registry:
            contrib_registry[arg].preload()
