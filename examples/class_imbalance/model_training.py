#!/usr/bin/env python

# # Class Imbalance Model Training Example
#
# This example trains a model utilizing a standard config, and then a config using oversampling

import logging
import shutil

# ## Import required libraries
from ludwig.api import LudwigModel
from ludwig.datasets import credit_card_fraud
from ludwig.visualize import learning_curves

# clean out old results
shutil.rmtree("./results", ignore_errors=True)
shutil.rmtree("./visualizations", ignore_errors=True)

# list models to train
list_of_model_ids = ["standard_model", "balanced_model"]
list_of_train_stats = []

training_set, _, _ = credit_card_fraud.load()

# ## Train models
for model_id in list_of_model_ids:
    print(">>>> training: ", model_id)

    # Define Ludwig model object that drive model training
    model = LudwigModel(config="./" + model_id + "_config.yaml", logging_level=logging.WARN)

    # initiate model training
    train_stats, _, _ = model.train(dataset=training_set, experiment_name="multiple_experiment", model_name=model_id)

    # save training stats for later use
    list_of_train_stats.append(train_stats)

    print(">>>>>>> completed: ", model_id, "\n")
