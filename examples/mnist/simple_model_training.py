#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#image-classification-mnist).
import logging
import shutil

import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)

# set up Python dictionary to hold model training parameters
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigModel(config,
                    logging_level=logging.INFO)

# load and split MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# initiate model training
(
    train_stats,  # training statistics
    _,
    output_directory  # location for training results saved to disk
) = model.train(
    training_set=training_set,
    test_set=test_set,
    experiment_name='simple_image_experiment',
    model_name='single_model',
    skip_save_processed_input=True
)
