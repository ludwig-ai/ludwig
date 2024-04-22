#! /usr/bin/env python
# Copyright (c) 2024 Predibase, Inc., 2019 Uber Technologies, Inc.
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
import argparse
import importlib

from ludwig.api_annotations import DeveloperAPI
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig


def pretrained_summary(model_name, **kwargs) -> None:
    module = importlib.import_module("torchvision.models")
    encoder_class = getattr(module, model_name)
    model = encoder_class()

    for name, _ in model.named_parameters():
        print(name)


@DeveloperAPI
def cli_summarize_pretrained(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script displays a summary of a pretrained model for freezing purposes.",
        prog="ludwig pretrained_summary",
        usage="%(prog)s [options]",
    )
    parser.add_argument("-m", "--model_name", help="output model layers", required=False, type=str)

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("pretrained_summary", *sys_argv)

    print_ludwig("Model Summary", LUDWIG_VERSION)
    pretrained_summary(**vars(args))
