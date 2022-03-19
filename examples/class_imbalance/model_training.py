#!/usr/bin/env python

# # Class Imbalance Model Training Example
#
# This example trains a model utilizing a standard config, and then a config using oversampling

import logging
import shutil

# Import required libraries
from ludwig.api import LudwigModel
from ludwig.datasets import imbalanced_insurance
from ludwig.visualize import compare_performance

# clean out old results
shutil.rmtree("./results", ignore_errors=True)
shutil.rmtree("./visualizations", ignore_errors=True)

# list models to train
list_of_model_ids = ["standard_model", "balanced_model"]
list_of_train_stats = []
list_of_eval_stats = []

training_set, val_set, test_set = imbalanced_insurance.load()

# Train models
for model_id in list_of_model_ids:
    print(">>>> training: ", model_id)

    # Define Ludwig model object that drive model training
    model = LudwigModel(config=model_id + "_config.yaml",
                        logging_level=logging.WARN)

    # initiate model training
    train_stats, _, _ = model.train(training_set=training_set,
                                    validation_set=val_set,
                                    test_set=test_set,
                                    experiment_name="balance_example",
                                    model_name=model_id,
                                    skip_save_model=True)

    # evaluate model on test_set
    eval_stats, _, _ = model.evaluate(test_set)

    # save eval stats for later use
    list_of_eval_stats.append(eval_stats)

    print(">>>>>>> completed: ", model_id, "\n")


compare_performance(
    list_of_eval_stats,
    "Response",
    model_names=list_of_model_ids,
    output_directory="./visualizations",
    file_format="png",
)
