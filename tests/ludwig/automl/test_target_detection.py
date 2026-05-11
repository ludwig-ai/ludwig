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
"""Unit tests for ludwig.automl.target_detection."""

import numpy as np
import pandas as pd
import pytest

from ludwig.automl.target_detection import (
    detect_all_target_candidates,
    detect_target_column,
    infer_task_type,
    TaskType,
)

# ---------------------------------------------------------------------------
# detect_target_column — name-based heuristics
# ---------------------------------------------------------------------------


def test_detect_by_name_target():
    df = pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": range(100),
            "target": [0, 1] * 50,
        }
    )
    result = detect_target_column(df)
    assert result.column == "target"
    assert result.confidence >= 0.90


def test_detect_by_name_label():
    df = pd.DataFrame(
        {
            "x": range(100),
            "label": ["cat", "dog"] * 50,
        }
    )
    result = detect_target_column(df)
    assert result.column == "label"


def test_detect_by_name_class():
    df = pd.DataFrame(
        {
            "x": range(50),
            "class": ["a", "b"] * 25,
        }
    )
    result = detect_target_column(df)
    assert result.column == "class"


# ---------------------------------------------------------------------------
# detect_target_column — positional heuristics
# ---------------------------------------------------------------------------


def test_detect_last_column():
    # No hint name; last column should be returned with confidence ~0.6
    df = pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": range(100),
            "outcome_col": [0, 1] * 50,
        }
    )
    result = detect_target_column(df)
    assert result.column == "outcome_col"
    assert abs(result.confidence - 0.60) < 0.05


# ---------------------------------------------------------------------------
# detect_target_column — binary balanced heuristic
# ---------------------------------------------------------------------------


def test_detect_binary_balanced():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(200),
            "flag": rng.choice([0, 1], size=200, p=[0.5, 0.5]),
        }
    )
    # flag is the last column, so it wins on the last-column heuristic at 0.6
    result = detect_target_column(df)
    assert result.column == "flag"


# ---------------------------------------------------------------------------
# infer_task_type
# ---------------------------------------------------------------------------


def test_infer_task_type_binary():
    series = pd.Series([0, 1, 0, 1, 1, 0])
    assert infer_task_type(series) == TaskType.BINARY


def test_infer_task_type_multiclass():
    series = pd.Series(["cat", "dog", "bird", "fish", "rabbit", "cat", "dog"])
    assert infer_task_type(series) == TaskType.MULTICLASS


def test_infer_task_type_regression():
    rng = np.random.default_rng(7)
    # >20 distinct continuous float values → REGRESSION
    series = pd.Series(rng.standard_normal(500))
    assert infer_task_type(series) == TaskType.REGRESSION


# ---------------------------------------------------------------------------
# detect_all_target_candidates
# ---------------------------------------------------------------------------


def test_all_candidates_sorted():
    df = pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": range(100),
            "target": [0, 1] * 50,  # should score highest (name hint)
        }
    )
    candidates = detect_all_target_candidates(df)
    assert len(candidates) > 0
    confidences = [c.confidence for c in candidates]
    assert confidences == sorted(confidences, reverse=True), "Candidates must be sorted by confidence descending"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_df_raises():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        detect_target_column(df)


def test_id_column_excluded():
    # Sequential 0..N integer column should not be the best candidate when a
    # better heuristic column (e.g. "target") is also present.
    df = pd.DataFrame(
        {
            "id": range(100),  # sequential int — ID column
            "feature": np.random.default_rng(0).standard_normal(100),
            "target": [0, 1] * 50,
        }
    )
    result = detect_target_column(df)
    assert result.column != "id", "Sequential ID column should not be chosen as the target"
