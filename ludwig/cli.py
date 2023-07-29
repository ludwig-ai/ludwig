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
import argparse
import sys

import ludwig.contrib
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logo


class CLI:
    """CLI describes a command line interface for interacting with Ludwig.

    Functions are described below.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="ludwig cli runner",
            usage=f"""\n{get_logo("ludwig cli", LUDWIG_VERSION)}
ludwig <command> [<args>]

Available sub-commands:
   train                 Trains a model
   predict               Predicts using a pretrained model
   evaluate              Evaluate a pretrained model's performance
   forecast              Forecast the next n data points in a timeseries using a pretrained model
   experiment            Runs a full experiment training a model and evaluating it
   hyperopt              Perform hyperparameter optimization
   benchmark             Run and track experiments on a number of datasets and configs, and export experiment artifacts.
   serve                 Serves a pretrained model
   visualize             Visualizes experimental results
   collect_summary       Prints names of weights and layers activations to use with other collect commands
   collect_weights       Collects tensors containing a pretrained model weights
   collect_activations   Collects tensors for each datapoint using a pretrained model
   datasets              Downloads and lists Ludwig-ready datasets
   export_torchscript    Exports Ludwig models to Torchscript
   export_triton         Exports Ludwig models to Triton
   export_neuropod       Exports Ludwig models to Neuropod
   export_mlflow         Exports Ludwig models to MLflow
   preprocess            Preprocess data and saves it into HDF5 and JSON format
   synthesize_dataset    Creates synthetic data for testing purposes
   init_config           Initialize a user config from a dataset and targets
   render_config         Renders the fully populated config with all defaults set
   check_install         Runs a quick training run on synthetic data to verify installation status
   upload                Push trained model artifacts to a registry (e.g., HuggingFace Hub)
""",
        )
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        from ludwig import train

        train.cli(sys.argv[2:])

    def predict(self):
        from ludwig import predict

        predict.cli(sys.argv[2:])

    def evaluate(self):
        from ludwig import evaluate

        evaluate.cli(sys.argv[2:])

    def forecast(self):
        from ludwig import forecast

        forecast.cli(sys.argv[2:])

    def experiment(self):
        from ludwig import experiment

        experiment.cli(sys.argv[2:])

    def hyperopt(self):
        from ludwig import hyperopt_cli

        hyperopt_cli.cli(sys.argv[2:])

    def benchmark(self):
        from ludwig.benchmarking import benchmark

        benchmark.cli(sys.argv[2:])

    def serve(self):
        from ludwig import serve

        serve.cli(sys.argv[2:])

    def visualize(self):
        from ludwig import visualize

        visualize.cli(sys.argv[2:])

    def collect_summary(self):
        from ludwig import collect

        collect.cli_collect_summary(sys.argv[2:])

    def collect_weights(self):
        from ludwig import collect

        collect.cli_collect_weights(sys.argv[2:])

    def collect_activations(self):
        from ludwig import collect

        collect.cli_collect_activations(sys.argv[2:])

    def export_torchscript(self):
        from ludwig import export

        export.cli_export_torchscript(sys.argv[2:])

    def export_triton(self):
        from ludwig import export

        export.cli_export_triton(sys.argv[2:])

    def export_neuropod(self):
        from ludwig import export

        export.cli_export_neuropod(sys.argv[2:])

    def export_mlflow(self):
        from ludwig import export

        export.cli_export_mlflow(sys.argv[2:])

    def preprocess(self):
        from ludwig import preprocess

        preprocess.cli(sys.argv[2:])

    def synthesize_dataset(self):
        from ludwig.data import dataset_synthesizer

        dataset_synthesizer.cli(sys.argv[2:])

    def init_config(self):
        from ludwig import automl

        automl.cli_init_config(sys.argv[2:])

    def render_config(self):
        from ludwig.utils import defaults

        defaults.cli_render_config(sys.argv[2:])

    def check_install(self):
        from ludwig import check

        check.cli(sys.argv[2:])

    def datasets(self):
        from ludwig import datasets

        datasets.cli(sys.argv[2:])

    def upload(self):
        from ludwig import upload

        upload.cli(sys.argv[2:])


def main():
    ludwig.contrib.preload(sys.argv)
    CLI()


if __name__ == "__main__":
    main()
