#!/usr/bin/env python

import logging
import os
import shutil

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income

shutil.rmtree("./results", ignore_errors=True)

model = LudwigModel(config="./config.yaml", logging_level=logging.INFO)

df = adult_census_income.load(split=False)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(dataset=df)

print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
