#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#image-classification-mnist).

# Import required libraries
import logging
import shutil

import yaml

from ludwig.api import LudwigPipeline

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)


# set up Python dictionary to hold model training parameters
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigPipeline(config,
                       logging_level=logging.INFO)


# initiate model training
(
    train_stats,  #training statistics
    _,
    output_directory  # location for training results saved to disk
) = model.train(
    training_set='./data/mnist_dataset_training.csv',
    test_set='./data/mnist_dataset_testing.csv',
    experiment_name='simple_image_experiment',
    model_name='single_model',
    skip_save_processed_input=True
)






