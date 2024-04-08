#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

LUDWIG_VERSION = "0.10.3.dev"

MODEL_FILE_NAME = "model"
MODEL_WEIGHTS_FILE_NAME = "model_weights"
MODEL_HYPERPARAMETERS_FILE_NAME = "model_hyperparameters.json"
TRAIN_SET_METADATA_FILE_NAME = "training_set_metadata.json"
TRAINING_PROGRESS_TRACKER_FILE_NAME = "training_progress.json"
TRAINING_CHECKPOINTS_DIR_PATH = "training_checkpoints"

TEST_STATISTICS_FILE_NAME = "test_statistics.json"

DESCRIPTION_FILE_NAME = "description.json"

PREDICTIONS_PARQUET_FILE_NAME = "predictions.parquet"
PREDICTIONS_SHAPES_FILE_NAME = "predictions.shapes.json"

TRAINING_PREPROC_FILE_NAME = "training.hdf5"

HYPEROPT_STATISTICS_FILE_NAME = "hyperopt_statistics.json"

CONFIG_YAML = "config.yaml"

DISABLE_PROGRESSBAR = False


def set_disable_progressbar(value):
    global DISABLE_PROGRESSBAR
    DISABLE_PROGRESSBAR = value


def is_progressbar_disabled():
    return DISABLE_PROGRESSBAR
