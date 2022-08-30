#!/usr/bin/env python

import copy
import logging
import shutil

import numpy as np
import yaml

import ludwig.visualize
from ludwig.api import LudwigModel
from ludwig.datasets import forest_cover

# clean out prior results
shutil.rmtree("./results_forest_cover", ignore_errors=True)
shutil.rmtree("./visualizations_forest_cover", ignore_errors=True)

# Download and prepare the dataset
dataset = forest_cover.load()

config_yaml = """
input_features:
  - name: Elevation
    type: number
  - name: Aspect
    type: number
  - name: Slope
    type: number
  - name: Horizontal_Distance_To_Hydrology
    type: number
  - name: Vertical_Distance_To_Hydrology
    type: number
  - name: Horizontal_Distance_To_Roadways
    type: number
  - name: Hillshade_9am
    type: number
  - name: Hillshade_Noon
    type: number
  - name: Hillshade_3pm
    type: number
  - name: Horizontal_Distance_To_Fire_Points
    type: number
  - name: Wilderness_Area
    type: category
  - name: Soil_Type
    type: category
output_features:
  - name: Cover_Type
    type: category
combiner:
  type: transformer
trainer:
  batch_size: 256
  learning_rate: .001
  epochs: 1
"""

uncalibrated_config = yaml.safe_load(config_yaml)

scaled_config = copy.deepcopy(uncalibrated_config)
scaled_config["output_features"][0]["calibration"] = True

uncalibrated_model = LudwigModel(config=uncalibrated_config, logging_level=logging.INFO)
uncalibrated_model.train(
    dataset,
    model_name="uncalibrated",
    experiment_name="forest_cover_calibration",
    output_directory="results_forest_cover",
)

scaled_model = LudwigModel(config=scaled_config, logging_level=logging.INFO)
scaled_model.train(
    dataset, model_name="scaled", experiment_name="forest_cover_calibration", output_directory="results_forest_cover"
)

# Generates predictions and performance statistics for the test set.
uncalibrated_test_stats, uncalibrated_test_predictions, _ = uncalibrated_model.evaluate(
    dataset, collect_predictions=True, collect_overall_stats=True
)

scaled_test_stats, scaled_test_predictions, _ = scaled_model.evaluate(
    dataset, collect_predictions=True, collect_overall_stats=True
)

uncalibrated_probs = np.stack(uncalibrated_test_predictions["Cover_Type_probabilities"], axis=0)
scaled_probs = np.stack(scaled_test_predictions["Cover_Type_probabilities"], axis=0)

ludwig.visualize.calibration_1_vs_all(
    probabilities_per_model=[uncalibrated_probs, scaled_probs],
    model_names=["Uncalibrated", "Calibrated"],
    ground_truth=dataset["Cover_Type"],
    metadata=uncalibrated_model.training_set_metadata,
    output_feature_name="Cover_Type",
    top_n_classes=[7, 7],
    labels_limit=7,
    output_directory="visualizations_forest_cover",
    file_format="png",
)
