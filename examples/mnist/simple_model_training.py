#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
# 
# This example is the API example for this Ludwig command line example
# (https://uber.github.io/ludwig/examples/#image-classification-mnist).

# Import required libraries

from ludwig.api import LudwigModel
import logging
import shutil
import yaml


# clean out prior results
try:
    shutil.rmtree('./results')
except FileNotFoundError:
    pass


# set up Python dictionary to hold model training parameters
with open('./model_definition.yaml','r') as f:
    model_definition = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigModel(model_definition,
                    logging_level=logging.INFO)

# initiate model training 
train_stats = model.train(data_train_csv='./data/mnist_dataset_training.csv',
                          data_test_csv='./data/mnist_dataset_testing.csv',
                         experiment_name='simple_image_experiment',
                         model_name='single_model')


model.close()





