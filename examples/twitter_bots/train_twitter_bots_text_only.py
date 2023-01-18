#!/usr/bin/env python
"""Trains twitter bots using tabular and text features only, no images."""
import logging
import os
import shutil

import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import twitter_bots
from ludwig.visualize import confusion_matrix, learning_curves

if __name__ == "__main__":
    # Cleans out prior results
    results_dir = os.path.join("outputs", "results")
    visualizations_dir = os.path.join("outputs", "visualizations")
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(visualizations_dir, ignore_errors=True)

    # Loads the dataset
    training_set, val_set, test_set = twitter_bots.load(split=True)

    config = yaml.safe_load(
        """
input_features:
  - name: created_at
    type: date
    column: created_at
  - name: default_profile
    type: binary
    column: default_profile
  - name: description
    type: text
    column: description
  - name: favourites_count
    type: number
    column: favourites_count
  - name: followers_count
    type: number
    column: followers_count
  - name: friends_count
    type: number
    column: friends_count
  - name: geo_enabled
    type: binary
    column: geo_enabled
  - name: lang
    type: category
    column: lang
  - name: location
    type: text
    column: location
  - name: screen_name
    type: text
    column: screen_name
  - name: statuses_count
    type: number
    column: statuses_count
  - name: verified
    type: binary
    column: verified
  - name: average_tweets_per_day
    type: number
    column: average_tweets_per_day
  - name: account_age_days
    type: number
    column: account_age_days
output_features:
  - name: account_type
    type: category
    column: account_type
trainer:
  batch_size: 16
defaults:
  text:
    preprocessing:
      tokenizer: space_punct
      max_sequence_length: 16
model_type: ecd
        """
    )

    model = LudwigModel(config, logging_level=logging.INFO)

    train_stats, preprocessed_data, output_directory = model.train(dataset=training_set, output_directory=results_dir)

    # Generates predictions and performance statistics for the test set.
    test_stats, predictions, output_directory = model.evaluate(
        test_set, collect_predictions=True, collect_overall_stats=True, output_directory=results_dir
    )

    confusion_matrix(
        [test_stats],
        model.training_set_metadata,
        "account_type",
        top_n_classes=[2],
        model_names=[""],
        normalize=True,
        output_directory=visualizations_dir,
        file_format="png",
    )

    # Visualizes learning curves, which show how performance metrics changed over time during training.
    learning_curves(
        train_stats, output_feature_name="account_type", output_directory=visualizations_dir, file_format="png"
    )
