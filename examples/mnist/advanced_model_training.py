#!/usr/bin/env python
# coding: utf-8

# # Multiple Model Training Example
# 
# This example trains multiple models and extracts training statistics

import glob
import logging
import os
import shutil
from collections import namedtuple

import yaml

# ## Import required libraries
from ludwig.api import LudwigModel
from ludwig.datasets import mnist
from ludwig.visualize import learning_curves

# clean out old results
shutil.rmtree('./results', ignore_errors=True)
shutil.rmtree('./visualizations', ignore_errors=True)

file_list = glob.glob('./data/*.json')
file_list += glob.glob('./data/*.hdf5')
for f in file_list:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# read in base config
with open('./config.yaml', 'r') as f:
    base_model = yaml.safe_load(f.read())

# Specify named tuple to keep track of training results
TrainingResult = namedtuple('TrainingResult', ['name', 'train_stats'])

# specify alternative architectures to test
FullyConnectedLayers = namedtuple('FullyConnectedLayers',
                                  ['name', 'fc_layers'])

list_of_fc_layers = [
    FullyConnectedLayers(name='Option1', fc_layers=[{'fc_size': 64}]),

    FullyConnectedLayers(name='Option2', fc_layers=[{'fc_size': 128},
                                                    {'fc_size': 64}]),

    FullyConnectedLayers(name='Option3', fc_layers=[{'fc_size': 128}])
]

#
list_of_train_stats = []

# load and split MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# ## Train models
for model_option in list_of_fc_layers:
    print('>>>> training: ', model_option.name)

    # set up Python dictionary to hold model training parameters
    config = base_model.copy()
    config['input_features'][0]['fc_layers'] = model_option.fc_layers
    config['training']['epochs'] = 5

    # Define Ludwig model object that drive model training
    model = LudwigModel(config,
                        logging_level=logging.INFO)

    # initiate model training
    train_stats, _, _ = model.train(
        training_set=training_set,
        test_set=test_set,
        experiment_name='multiple_experiment',
        model_name=model_option.name)

    # save training stats for later use
    list_of_train_stats.append(TrainingResult(name=model_option.name, train_stats=train_stats))

    print('>>>>>>> completed: ', model_option.name, '\n')


# generating learning curves from training
option_names = [trs.name for trs in list_of_train_stats]
train_stats = [trs.train_stats for trs in list_of_train_stats]
learning_curves(train_stats, 'Survived',
                model_names=option_names,
                output_directory='./visualizations',
                file_format='png')



