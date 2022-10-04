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

import numpy as np

from ludwig.backend import Backend
from ludwig.models.ecd import ECD


class Calibrator:
    """Calibrator calibrates the output probabilities of a model."""

    def __init__(self, model: ECD, backend: Backend, batch_size: int = 128):
        self.model = model
        self.backend = backend
        self.batch_size = batch_size

    def calibration_enabled(self):
        """Calibration is enabled if the config requests calibration for any output feature.

        If no output features have calibration enabled, the calibration phase should be skipped.
        """
        return any(o.calibration_module is not None for o in self.model.output_features.values())

    def train_calibration(self, dataset, dataset_name: str):
        """Calibrates model output probabilities on validation set after training.

        This works well for most datasets, though it may fail for some difficult or extremely imbalanced datasets.
        """
        if not self.calibration_enabled():
            # Early out if no output features have calibration enabled.
            return
        with self.backend.create_predictor(self.model, batch_size=self.batch_size) as predictor:
            metrics, predictions = predictor.batch_evaluation(
                dataset, collect_predictions=True, collect_logits=True, dataset_name=dataset_name
            )

        dataset_df = dataset.to_df()
        for output_feature in self.model.output_features.values():
            if output_feature.calibration_module is not None:
                feature_logits_key = f"{output_feature.feature_name}_logits"
                if feature_logits_key in predictions:
                    feature_logits = self.backend.df_engine.compute(predictions[feature_logits_key])
                    feature_labels = self.backend.df_engine.compute(dataset_df[output_feature.proc_column])
                    output_feature.calibration_module.train_calibration(
                        np.stack(feature_logits.values, axis=0), np.stack(feature_labels.values, axis=0)
                    )
