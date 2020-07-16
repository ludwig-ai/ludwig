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
"""

import sys

from .contribs import contrib_registry


def use_contrib(name, *args, **kwargs):
    # Import a contrib package and cache its instance, if appropriate
    contrib_class = contrib_registry["classes"][name]
    if contrib_class not in [obj.__class__ for obj in
                             contrib_registry["instances"]]:
        try:
            instance = contrib_class.import_call(*args, **kwargs)
        except Exception:
            instance = None

        # Save instance in registry
        if instance:
            contrib_registry["instances"].append(instance)


def contrib_import():
    """
    Checks for contrib flags, and calls static method:

    ContribClass.import_call(argv_list)

    import_call() will return an instance to the class
    if appropriate (all dependencies are met, for example).
    """
    argv_list = sys.argv
    argv_set = set(argv_list)
    for contrib_name in contrib_registry["classes"]:
        parameter_name = '--' + contrib_name
        if parameter_name in argv_set:
            use_contrib(contrib_name, *argv_list)

            # Clean up and remove the flag
            sys.argv.remove(parameter_name)


def contrib_command(command, *args, **kwargs):
    """
    If a contrib has an instance in the registry,
    this will call:

    ContribInstance.COMMAND(*args, **kwargs)
    """
    for instance in contrib_registry["instances"]:
        method = getattr(instance, command, None)
        if method:
            method(*args, **kwargs)
