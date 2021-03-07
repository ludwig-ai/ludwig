#!/usr/bin/env python
# coding: utf-8

# # Multiple Model Training Example
# 
# This example trains multiple models and extracts training statistics

import logging
import shutil

# ## Import required libraries
from ludwig.api import LudwigModel
from ludwig.datasets import titanic
from ludwig.visualize import learning_curves

# clean out old results
shutil.rmtree('./results', ignore_errors=True)
shutil.rmtree('./visualizations', ignore_errors=True)

# list models to train
list_of_model_ids = ['model1', 'model2']
list_of_train_stats = []

training_set, _, _ = titanic.load(split=True)

# ## Train models
for model_id in list_of_model_ids:
    print('>>>> training: ', model_id)

    # Define Ludwig model object that drive model training
    model = LudwigModel(config='./' + model_id + '_config.yaml',
                        logging_level=logging.WARN)

    # initiate model training
    train_stats, _, _ = model.train(
        dataset=training_set,
        experiment_name='multiple_experiment',
        model_name=model_id
    )

    # save training stats for later use
    list_of_train_stats.append(train_stats)

    print('>>>>>>> completed: ', model_id,'\n')

# generating learning curves from training
learning_curves(list_of_train_stats, 'Survived',
                model_names=list_of_model_ids,
                output_directory='./visualizations',
                file_format='png')



