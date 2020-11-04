#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#image-classification-mnist).


import logging
import shutil
import tempfile
import yaml

from ludwig.api import LudwigModel
from ludwig.datasets.mnist import Mnist

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)


# set up Python dictionary to hold model training parameters
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigModel(config,
                    logging_level=logging.INFO)


class MnistDataset(Mnist):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)


def main():
    SPLIT = "split"
    with tempfile.TemporaryDirectory() as tmpdir:
        mnist = MnistDataset(tmpdir)
        dataset_df = mnist.load()
        training_set = dataset_df[dataset_df[SPLIT] == "0"]
        test_set = dataset_df[dataset_df[SPLIT] == "2"]
        model.train(training_set=training_set, test_set=test_set)


if __name__ == "__main__":
    main()







