#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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

import os

import numpy as np
import torch

from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.ecd import ECD
from ludwig.models.predictor import Predictor


class Calibrator:
    """Calibrator calibrates the output probabilities of a model."""

    def __init__(self, model: ECD, batch_size: int = 128, horovod=None, skip_save_model=False):
        self.model = model
        self.batch_size = batch_size
        self.horovod = horovod
        self.skip_save_model = skip_save_model

    def calibration(self, dataset, dataset_name: str, save_path: str):
        """Calibrates model output probabilities on validation set after training.

        This works well for most datasets, though it may fail for some difficult or extremely imbalanced datasets.
        """
        if all(o.calibration_module is None for o in self.model.output_features.values()):
            # Early out if no output features have calibration enabled.
            return
        predictor = Predictor(self.model, batch_size=self.batch_size, horovod=self.horovod)
        metrics, predictions = predictor.batch_evaluation(
            dataset, collect_predictions=True, collect_logits=True, collect_labels=True, dataset_name=dataset_name
        )
        for output_feature in self.model.output_features.values():
            feature_logits_key = "%s_logits" % output_feature.feature_name
            if feature_logits_key in predictions:
                feature_logits = predictions[feature_logits_key]
                feature_labels = predictions["%s_labels" % output_feature.feature_name]
                output_feature.calibrate(
                    np.stack(feature_logits.values, axis=0), np.stack(feature_labels.values, axis=0)
                )
        if not self.skip_save_model:
            model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
            torch.save(self.model.state_dict(), model_weights_path)
