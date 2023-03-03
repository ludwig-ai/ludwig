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

import argparse

from ludwig.contribs import contrib_registry, ContribLoader


def create_load_action(contrib_loader: ContribLoader) -> argparse.Action:
    class LoadContribAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            items = getattr(namespace, self.dest) or []
            items.append(contrib_loader.load())
            setattr(namespace, self.dest, items)

    return LoadContribAction


def add_contrib_callback_args(parser: argparse.ArgumentParser):
    for contrib_name, contrib_loader in contrib_registry.items():
        parser.add_argument(
            f"--{contrib_name}",
            dest="callbacks",
            nargs=0,
            action=create_load_action(contrib_loader),
        )


def preload(argv):
    for arg in argv:
        if arg.startswith("--"):
            arg = arg[2:]

        if arg in contrib_registry:
            contrib_registry[arg].preload()
