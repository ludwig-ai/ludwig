#!/usr/bin/env python

import logging
import os
import shutil

import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import twitter_bots
from ludwig.utils.fs_utils import rename
from ludwig.visualize import confusion_matrix, learning_curves

if __name__ == "__main__":
    # Cleans out prior results
    shutil.rmtree("./results", ignore_errors=True)
    shutil.rmtree(".visualizations", ignore_errors=True)

    # Loads the dataset
    dataset = twitter_bots.TwitterBots(cache_dir=".")
    training_set, val_set, test_set = dataset.load(split=True)
    # Moves profile images into local directory, so relative paths in the dataset will be resolved.
    rename(os.path.join(dataset.processed_dataset_path, "profile_images"), "./profile_images")

    with open("./config.yaml") as f:
        config = yaml.safe_load(f.read())

    model = LudwigModel(config, logging_level=logging.INFO)

    train_stats, preprocessed_data, output_directory = model.train(dataset=training_set)

    # Generates predictions and performance statistics for the test set.
    test_stats, predictions, output_directory = model.evaluate(
        test_set, collect_predictions=True, collect_overall_stats=True
    )

    confusion_matrix(
        [test_stats],
        model.training_set_metadata,
        "account_type",
        top_n_classes=[2],
        model_names=[""],
        normalize=True,
        output_directory="./visualizations",
        file_format="png",
    )

    # Visualizes learning curves, which show how performance metrics changed over time during training.
    learning_curves(
        train_stats, output_feature_name="account_type", output_directory="./visualizations", file_format="png"
    )
