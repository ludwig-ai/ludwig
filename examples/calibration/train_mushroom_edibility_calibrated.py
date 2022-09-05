#!/usr/bin/env python

import copy
import logging
import shutil

import numpy as np
import yaml

import ludwig.visualize
from ludwig.api import LudwigModel
from ludwig.datasets import mushroom_edibility

# clean out prior results
shutil.rmtree("./results_mushroom_edibility", ignore_errors=True)
shutil.rmtree("./visualizations_mushroom_edibility", ignore_errors=True)

# Download and prepare the dataset
dataset = mushroom_edibility.load()

# This dataset has no split, so add a split column
dataset.split = np.random.choice(3, len(dataset), p=(0.7, 0.1, 0.2))

config_yaml = """
input_features:
  - name: cap-shape
    type: category
  - name: cap-surface
    type: category
  - name: cap-color
    type: category
  - name: bruises?
    type: category
  - name: odor
    type: category
  - name: gill-attachment
    type: category
  - name: gill-spacing
    type: category
  - name: gill-size
    type: category
  - name: gill-color
    type: category
  - name: stalk-shape
    type: category
  - name: stalk-root
    type: category
  - name: stalk-surface-above-ring
    type: category
  - name: stalk-surface-below-ring
    type: category
  - name: stalk-color-above-ring
    type: category
  - name: stalk-color-below-ring
    type: category
  - name: veil-type
    type: category
  - name: veil-color
    type: category
  - name: ring-number
    type: category
  - name: ring-type
    type: category
  - name: spore-print-color
    type: category
  - name: population
    type: category
  - name: habitat
    type: category
output_features:
  - name: class
    type: category
combiner:
  type: concat
trainer:
  batch_size: 256
  learning_rate: .0001
  epochs: 10
"""

uncalibrated_config = yaml.safe_load(config_yaml)

scaled_config = copy.deepcopy(uncalibrated_config)
scaled_config["output_features"][0]["calibration"] = True

uncalibrated_model = LudwigModel(config=uncalibrated_config, logging_level=logging.INFO)
uncalibrated_model.train(
    dataset,
    model_name="uncalibrated",
    experiment_name="mushroom_edibility_calibration",
    output_directory="results_mushroom_edibility",
)

scaled_model = LudwigModel(config=scaled_config, logging_level=logging.INFO)
scaled_model.train(
    dataset,
    model_name="scaled",
    experiment_name="mushroom_edibility_calibration",
    output_directory="results_mushroom_edibility",
)

# Generates predictions and performance statistics for the test set.
uncalibrated_test_stats, uncalibrated_test_predictions, _ = uncalibrated_model.evaluate(
    dataset, collect_predictions=True, collect_overall_stats=True
)

scaled_test_stats, scaled_test_predictions, _ = scaled_model.evaluate(
    dataset, collect_predictions=True, collect_overall_stats=True
)

uncalibrated_probs = np.stack(uncalibrated_test_predictions["class_probabilities"], axis=0)
scaled_probs = np.stack(scaled_test_predictions["class_probabilities"], axis=0)

ludwig.visualize.calibration_1_vs_all(
    probabilities_per_model=[uncalibrated_probs, scaled_probs],
    model_names=["Uncalibrated", "Calibrated"],
    ground_truth=dataset["class"],
    metadata=uncalibrated_model.training_set_metadata,
    output_feature_name="class",
    top_n_classes=[3, 3],
    labels_limit=3,
    output_directory="visualizations_mushroom_edibility",
    file_format="png",
)
