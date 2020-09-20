#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#image-classification-mnist).

# Import required libraries
import os
import logging
import shutil

import yaml
import pandas as pd

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)


# set up Python dictionary to hold model training parameters
with open('./model_definition.yaml','r') as f:
    model_definition = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigModel(model_definition,
                    logging_level=logging.INFO)

current_working_directory = os.getcwd()

# read in data set and randomize sequence of records
train_df = pd.read_csv('./data/mnist_dataset_training.csv')
train_df['image_path'] = current_working_directory + '/data/' + train_df['image_path']
train_df = train_df.sample(train_df.shape[0])

test_df = pd.read_csv('./data/mnist_dataset_testing.csv')
test_df['image_path'] = current_working_directory + '/data/' + test_df['image_path']

# initiate model training
(
    train_stats,  #training statistics
    _,
    output_directory  # location for training results saved to disk
) = model.train(
    training_set=train_df,
    test_set=test_df,
    experiment_name='simple_image_experiment',
    model_name='single_model',
    skip_save_processed_input=True
)






