#! /usr/bin/env python
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
import argparse
import sys

import ludwig.contrib

ludwig.contrib.contrib_import()


class CLI(object):
    """CLI describes a command line interface for interacting with Ludwig, there
    are several different functions that can be performed. These functions are:
    - experiment - run an experiment using ludwig
    - predict - Given a list of $hat{y}$ values, compute $d(\\hat{y}, y) under a
      specified metric
    - train - trains a model on the input file specified to it
    - visualize - Analysis of the results for the model on the dataset and
      presents a variety of plots to understand and evaluate the results
    - collect_weights - Collects the weights for a pretrained model as a tensor
      representation
    - collect_activations - For each datapoint, there exists a corresponding
      tensor representation which are collected through this method
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='ludwig cli runner',
            usage='''ludwig <command> [<args>]

Available sub-commands:
   experiment            Runs a full experiment training a model and testing it
   train                 Trains a model
   predict               Predicts using a pretrained model
   serve                 Serves a pretrained model
   test                  Tests a pretrained model
   visualize             Visualizes experimental results
   collect_weights       Collects tensors containing a pretrained model weights
   collect_activations   Collects tensors for each datapoint using a pretrained model
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def experiment(self):
        from ludwig import experiment
        ludwig.contrib.contrib_command("experiment", *sys.argv)
        experiment.cli(sys.argv[2:])

    def train(self):
        from ludwig import train
        ludwig.contrib.contrib_command("train", *sys.argv)
        train.cli(sys.argv[2:])

    def predict(self):
        from ludwig import predict
        ludwig.contrib.contrib_command("predict", *sys.argv)
        predict.cli(sys.argv[2:])

    def serve(self):
        from ludwig import serve
        ludwig.contrib.contrib_command("serve", *sys.argv)
        serve.cli(sys.argv[2:])

    def test(self):
        from ludwig import test_performance
        ludwig.contrib.contrib_command("test", *sys.argv)
        test_performance.cli(sys.argv[2:])

    def visualize(self):
        from ludwig import visualize
        ludwig.contrib.contrib_command("visualize", *sys.argv)
        visualize.cli(sys.argv[2:])

    def collect_weights(self):
        from ludwig import collect
        ludwig.contrib.contrib_command("collect_weights", *sys.argv)
        collect.cli_collect_weights(sys.argv[2:])

    def collect_activations(self):
        from ludwig import collect
        ludwig.contrib.contrib_command("collect_activations", *sys.argv)
        collect.cli_collect_activations(sys.argv[2:])


def main():
    CLI()


if __name__ == '__main__':
    main()
