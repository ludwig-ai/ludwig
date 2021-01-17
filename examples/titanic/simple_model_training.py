#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#kaggles-titanic-predicting-survivors).

# Import required libraries
import logging
import os
import shutil

from ludwig.api import LudwigModel
from ludwig.datasets import titanic

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)

# Download and prepare the dataset
training_set, _, _ = titanic.load(split=True)

# Define Ludwig model object that drive model training
model = LudwigModel(config='./model1_config.yaml',
                    logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory # location of training results stored on disk
 ) = model.train(
    dataset=training_set,
    experiment_name='simple_experiment',
    model_name='simple_model'
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)





