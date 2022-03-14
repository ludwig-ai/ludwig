#!/usr/bin/env python

# # Class Imbalance Model Training Example
#
# This example trains a model utilizing a standard config, and then a config using oversampling

import logging
import shutil

# Import required libraries
from ludwig.api import LudwigModel
from ludwig.datasets import imbalanced_insurance
from ludwig.visualize import learning_curves

# clean out old results
shutil.rmtree("./results", ignore_errors=True)
shutil.rmtree("./visualizations", ignore_errors=True)

# list models to train
list_of_model_ids = ["standard_model", "balanced_model"]
list_of_train_stats = []

training_set = imbalanced_insurance.load()

config = {
    "type": "local",
}

# Train models
for model_id in list_of_model_ids:
    print(">>>> training: ", model_id)

    # Define Ludwig model object that drive model training
    model = LudwigModel(config="examples/class_imbalance/" + model_id + "_config.yaml", logging_level=logging.WARN,
                        backend=config)

    # initiate model training
    train_stats, _, _ = model.train(dataset=training_set, experiment_name="imbalance_experiment", model_name=model_id) # TODO: Specify output directory

    # save training stats for later use
    list_of_train_stats.append(train_stats)

    print(">>>>>>> completed: ", model_id, "\n")

learning_curves(
    list_of_train_stats,
    "Class",
    model_names=list_of_model_ids,
    output_directory="./visualizations",
    file_format="png",
)