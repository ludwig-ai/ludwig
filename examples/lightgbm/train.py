#!/usr/bin/env python
# coding: utf-8

import logging
import os
import shutil

from ludwig.api import LudwigModel

shutil.rmtree('./results', ignore_errors=True)

model = LudwigModel(config='./config.yaml',
                    logging_level=logging.INFO)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory # location of training results stored on disk
 ) = model.train(
    dataset='/Users/tgaddair/data/uci-income/train.csv',
    skip_save_model=True,
)

print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
