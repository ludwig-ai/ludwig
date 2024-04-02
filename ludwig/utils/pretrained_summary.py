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

models = [
    "alexnet",
    "convnext",
    "convnext_base",
    "convnext_large",
    "convnext_small",
    "convnext_tiny",
    "densenet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "efficientnet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
    "googlenet",
    "inception",
    "inception_v3",
    "maxvit",
    "maxvit_t",
    "mnasnet",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
    "mobilenet",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mobilenetv2",
    "mobilenetv3",
    "regnet",
    "regnet_x_16gf",
    "regnet_x_1_6gf",
    "regnet_x_32gf",
    "regnet_x_3_2gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_8gf",
    "regnet_y_128gf",
    "regnet_y_16gf",
    "regnet_y_1_6gf",
    "regnet_y_32gf",
    "regnet_y_3_2gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_8gf",
    "resnet",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext50_32x4d",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "shufflenetv2",
    "squeezenet",
    "squeezenet1_0",
    "squeezenet1_1",
    "swin_transformer",
    "vgg",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "vit_b_16",
    "vit_b_32",
    "vit_h_14",
    "vit_l_16",
    "vit_l_32",
    "wide_resnet101_2",
    "wide_resnet50_2",
]


def pretrained_summary(model_name, **kwargs) -> None:
    if model_name in models:
        module = importlib.import_module("torchvision.models")
        encoder_class = getattr(module, model_name)
        model = encoder_class()

        for name, _ in model.named_parameters():
            print(name)
    else:
        print(f"No encoder found for '{model_name}'")


@DeveloperAPI
def cli_summarize_pretrained(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script displays a summary of a pretrained model for freezing purposes.",
        prog="ludwig pretrained_summary",
        usage="%(prog)s [options]",
    )
    parser.add_argument("-m", "--model_name", help="output model layers", required=False, type=str)
    parser.add_argument("-l", "--list_models", action="store_true", help="print available models")

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("pretrained_summary", *sys_argv)

    print_ludwig("Model Summary", LUDWIG_VERSION)
    if args.list_models:
        print("Available models:")
        for model in models:
            print(f"- {model}")
    else:
        pretrained_summary(**vars(args))
