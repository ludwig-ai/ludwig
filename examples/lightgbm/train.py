#!/usr/bin/env python

import logging
import os
import shutil

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income

shutil.rmtree("./results", ignore_errors=True)

model = LudwigModel(config="./config.yaml", logging_level=logging.INFO)

train, val, test = adult_census_income.load(split=True)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    training_set=train,
    validation_set=val,
    test_set=test,
)

print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
