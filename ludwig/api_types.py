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
"""Lightweight data-class types shared across Ludwig's public API.

These are intentionally kept in a separate module so callers can import
``EvaluationFrequency``, ``TrainingStats``, ``PreprocessedDataset``, and
``TrainingResults`` without pulling in the entire ``ludwig.api`` module
(which transitively imports PyTorch, the full model registry, etc.).

All four types remain importable from ``ludwig.api`` for backward compatibility.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

from ludwig.api_annotations import PublicAPI
from ludwig.constants import TEST, TRAINING, VALIDATION
from ludwig.data.dataset.base import Dataset
from ludwig.types import TrainingSetMetadataDict


@PublicAPI
@dataclass
class EvaluationFrequency:
    """Represents the frequency of periodic evaluation of a metric during training. For example:

    "every epoch"
    frequency: 1, period: EPOCH

    "every 50 steps".
    frequency: 50, period: STEP
    """

    frequency: float = 1.0
    period: str = "epoch"  # One of "epoch" or "step".

    EPOCH: ClassVar[str] = "epoch"  # One epoch is a single pass through the training set.
    STEP: ClassVar[str] = "step"  # One step is training on one mini-batch.


@PublicAPI
@dataclass
class TrainingStats:
    """Training statistics for all splits (training, validation, test)."""

    training: dict[str, Any]
    validation: dict[str, Any]
    test: dict[str, Any]
    evaluation_frequency: EvaluationFrequency = dataclasses.field(default_factory=EvaluationFrequency)

    def __contains__(self, key: object) -> bool:
        return (
            (key == TRAINING and self.training)
            or (key == VALIDATION and self.validation)
            or (key == TEST and self.test)
        )

    def __getitem__(self, key: str) -> dict[str, Any]:
        return {TRAINING: self.training, VALIDATION: self.validation, TEST: self.test}[key]

    # Make TrainingStats a proper Mapping so dict(ts) and generic helpers like
    # ludwig.utils.numerical_test_utils.assert_all_finite treat it as a dict
    # rather than falling back to integer-index iteration (KeyError(0)).
    _KEYS = (TRAINING, VALIDATION, TEST)

    def keys(self) -> tuple[str, ...]:
        return self._KEYS

    def __iter__(self):
        return iter(self._KEYS)


@PublicAPI
@dataclass
class PreprocessedDataset:
    training_set: Dataset
    validation_set: Dataset
    test_set: Dataset
    training_set_metadata: TrainingSetMetadataDict

    def __iter__(self):
        import warnings

        warnings.warn(
            "Tuple unpacking of PreprocessedDataset is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self.training_set, self.validation_set, self.test_set, self.training_set_metadata))

    def __getitem__(self, index):
        import warnings

        warnings.warn(
            "Indexed access of PreprocessedDataset is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self.training_set, self.validation_set, self.test_set, self.training_set_metadata)[index]


@PublicAPI
@dataclass
class TrainingResults:
    train_stats: TrainingStats
    preprocessed_data: PreprocessedDataset
    output_directory: str

    def __iter__(self):
        import warnings

        warnings.warn(
            "Tuple unpacking of TrainingResults is deprecated. "
            "Use attribute access instead: result.train_stats, result.preprocessed_data, result.output_directory",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self.train_stats, self.preprocessed_data, self.output_directory))

    def __getitem__(self, index):
        import warnings

        warnings.warn(
            "Indexed access of TrainingResults is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self.train_stats, self.preprocessed_data, self.output_directory)[index]
