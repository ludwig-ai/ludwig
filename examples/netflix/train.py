#!/usr/bin/env python
# coding: utf-8

import logging
import os
import shutil

from ludwig.api import LudwigModel

from ludwig.data.dataset.parquet import ParquetDataset
from ludwig.utils.misc_utils import get_features

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)

# Define Ludwig model object that drive model training
model = LudwigModel(config='./model_config.yaml',
                    logging_level=logging.INFO)

# (
#     training_set,
#     validation_set,
#     test_set,
#     training_set_metadata
# ) = model.preprocess(
#     dataset='./data/dataset.parquet/part.0.parquet',
#     experiment_name='netflix_experiment',
#     model_name='netflix_model',
#     skip_save_processed_input=False,
# )


features = get_features(model.config)
train_dataset = ParquetDataset('file://' + os.path.abspath('./data/dataset.train.parquet'), features)
val_dataset = ParquetDataset('file://' + os.path.abspath('./data/dataset.train.parquet'), features)
test_dataset = ParquetDataset('file://' + os.path.abspath('./data/dataset.train.parquet'), features)

train_stats, preprocessed_data, output_directory= model.train(
    training_set=train_dataset,
    validation_set=val_dataset,
    test_set=test_dataset
)

print('FINISHED: ', train_stats)

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
