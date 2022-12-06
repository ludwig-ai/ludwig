#! /usr/bin/env python
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
from collections import OrderedDict
from pprint import pformat
from typing import Dict, Union

from ludwig.api_annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
def get_logging_level_registry() -> Dict[str, int]:
    return {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET,
    }


@DeveloperAPI
def get_logo(message, ludwig_version):
    return "\n".join(
        [
            "███████████████████████",
            "█ █ █ █  ▜█ █ █ █ █   █",
            "█ █ █ █ █ █ █ █ █ █ ███",
            "█ █   █ █ █ █ █ █ █ ▌ █",
            "█ █████ █ █ █ █ █ █ █ █",
            "█     █  ▟█     █ █   █",
            "███████████████████████",
            f"ludwig v{ludwig_version} - {message}",
            "",
        ]
    )


@DeveloperAPI
def print_ludwig(message, ludwig_version):
    logger.info(get_logo(message, ludwig_version))


@DeveloperAPI
def print_boxed(text, print_fun=logger.info):
    box_width = len(text) + 2
    print_fun("")
    print_fun("╒{}╕".format("═" * box_width))
    print_fun(f"│ {text.upper()} │")
    print_fun("╘{}╛".format("═" * box_width))
    print_fun("")


@DeveloperAPI
def repr_ordered_dict(d: OrderedDict):
    return "{" + ",\n  ".join(f"{x}: {pformat(y, indent=4)}" for x, y in d.items()) + "}"


@DeveloperAPI
def query_yes_no(question: str, default: Union[str, None] = "yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    Args:
        question: String presented to the user
        default: The presumed answer from the user. Must be "yes", "no", or None (Answer is required)

    Returns: Boolean based on prompt response
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        logger.info(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            logger.info("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
