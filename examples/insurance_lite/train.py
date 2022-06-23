#!/usr/bin/env python

# # Simple Model Training Example on multi-modal data.

# Import required libraries
import logging
import os
import shutil

from ludwig.api import LudwigModel
from ludwig.datasets import insurance_lite

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
dataset = insurance_lite.load()

# Define Ludwig model object that drive model training
model = LudwigModel(config="./config.yaml", logging_level=logging.INFO, backend="local")


# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(dataset=dataset, experiment_name="simple_experiment", model_name="simple_model")

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
