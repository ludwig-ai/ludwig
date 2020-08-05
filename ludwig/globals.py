#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
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
from ludwig.utils.horovod_utils import should_use_horovod

LUDWIG_VERSION = '0.3.0-a.2'

MODEL_WEIGHTS_FILE_NAME = 'model_weights'
MODEL_HYPERPARAMETERS_FILE_NAME = 'model_hyperparameters.json'
TRAIN_SET_METADATA_FILE_NAME = 'train_set_metadata.json'
TRAINING_PROGRESS_TRACKER_FILE_NAME = 'training_progress.json'
TRAINING_CHECKPOINTS_DIR_PATH = 'training_checkpoints'

DISABLE_PROGRESSBAR = False

ON_MASTER = True


def set_disable_progressbar(value):
    global DISABLE_PROGRESSBAR
    DISABLE_PROGRESSBAR = value


def is_progressbar_disabled():
    return DISABLE_PROGRESSBAR


def set_on_master(use_horovod):
    global ON_MASTER
    if should_use_horovod(use_horovod):
        try:
            import horovod.tensorflow
            horovod.tensorflow.init()
            ON_MASTER = horovod.tensorflow.rank() == 0
        except ImportError:
            raise ValueError("use_horovod parameter specified, "
                             "but cannot import horovod.tensorflow. "
                             "Install horovod following the instructions at: "
                             " https://github.com/horovod/horovod")
    else:
        ON_MASTER = True


def is_on_master():
    return ON_MASTER
