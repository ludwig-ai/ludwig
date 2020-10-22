#!/usr/bin/env python
# coding: utf-8

import logging
import os
import shutil

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)

# Define Ludwig model object that drive model training
model = LudwigModel(config='./model_config.yaml',
                    logging_level=logging.INFO)

(
    training_set,
    validation_set,
    test_set,
    training_set_metadata
) = model.preprocess(
    dataset='./data/dataset.parquet/part.0.parquet',
    experiment_name='netflix_experiment',
    model_name='netflix_model'
)

print(training_set)

# # initiate model training
# (
#     train_stats,  # dictionary containing training statistics
#     preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
#     output_directory # location of training results stored on disk
#  ) = model.train(
#     dataset='./data/dataset.parquet',
#     experiment_name='netflix_experiment',
#     model_name='netflix_model'
# )

# # list contents of output directory
# print("contents of output directory:", output_directory)
# for item in os.listdir(output_directory):
#     print("\t", item)
