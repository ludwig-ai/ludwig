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
"""Unit tests for ludwig.utils.dataset_quality."""

import numpy as np
import pandas as pd

from ludwig.utils.dataset_quality import (
    check_dataset_quality,
    CheckStatus,
    drop_quality_issues,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.integers(0, 5, size=n).astype(float),
            "feature_c": rng.standard_normal(n),
            "label": rng.choice(["cat", "dog"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_dataset_passes():
    df = _clean_df(500)
    report = check_dataset_quality(df, target_column="label", dataset_name="test", min_rows=200)
    assert report.passed


def test_too_few_rows_fails():
    df = _clean_df(50)
    report = check_dataset_quality(df, target_column="label", min_rows=200)
    failure_names = [c.name for c in report.failures]
    assert "minimum_size" in failure_names


def test_too_few_features_fails():
    # Only 1 non-target column
    df = pd.DataFrame({"feature_a": range(300), "label": [0, 1] * 150})
    report = check_dataset_quality(df, target_column="label", min_features=2)
    failure_names = [c.name for c in report.failures]
    assert "minimum_features" in failure_names


def test_constant_column_warns():
    df = _clean_df(300)
    df["constant"] = 42  # all same value
    report = check_dataset_quality(df, target_column="label")
    check_statuses = {c.name: c.status for c in report.checks}
    assert check_statuses["constant_columns"] == CheckStatus.WARN


def test_near_duplicate_warns():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    df = pd.DataFrame(
        {
            "a": x,
            "b": x,  # perfectly correlated with a
            "label": rng.choice([0, 1], size=300),
        }
    )
    report = check_dataset_quality(df, target_column="label")
    check_statuses = {c.name: c.status for c in report.checks}
    assert check_statuses["near_duplicate_columns"] == CheckStatus.WARN


def test_target_leakage_fails():
    rng = np.random.default_rng(2)
    target = rng.standard_normal(300)
    leaky = target + rng.standard_normal(300) * 0.001  # r > 0.99
    df = pd.DataFrame(
        {
            "leaky_feature": leaky,
            "other": rng.standard_normal(300),
            "target": target,
        }
    )
    report = check_dataset_quality(df, target_column="target")
    failure_names = [c.name for c in report.failures]
    assert "target_leakage" in failure_names


def test_id_column_warns():
    df = pd.DataFrame(
        {
            "id": range(300),  # sequential integer
            "feature": np.random.default_rng(3).standard_normal(300),
            "label": [0, 1] * 150,
        }
    )
    report = check_dataset_quality(df, target_column="label")
    check_statuses = {c.name: c.status for c in report.checks}
    assert check_statuses["id_columns"] == CheckStatus.WARN


def test_class_imbalance_warns():
    rng = np.random.default_rng(4)
    n = 10_000
    # Minority class is 0.1% of rows (10 samples out of 10000)
    labels = np.array([0] * (n - 10) + [1] * 10)
    df = pd.DataFrame(
        {
            "feature": rng.standard_normal(n),
            "label": labels,
        }
    )
    report = check_dataset_quality(df, target_column="label")
    check_statuses = {c.name: c.status for c in report.checks}
    assert check_statuses["class_imbalance"] == CheckStatus.WARN


def test_single_class_fails():
    df = pd.DataFrame(
        {
            "feature": range(300),
            "label": ["cat"] * 300,  # only 1 distinct value
        }
    )
    report = check_dataset_quality(df, target_column="label")
    failure_names = [c.name for c in report.failures]
    assert "single_class" in failure_names


def test_drop_quality_issues_removes_constant():
    df = _clean_df(300)
    df["constant_col"] = 7
    report = check_dataset_quality(df, target_column="label")
    cleaned = drop_quality_issues(df, report)
    assert "constant_col" not in cleaned.columns


def test_drop_quality_issues_removes_id():
    df = pd.DataFrame(
        {
            "id_col": range(300),
            "feature": np.random.default_rng(5).standard_normal(300),
            "label": [0, 1] * 150,
        }
    )
    report = check_dataset_quality(df, target_column="label")
    cleaned = drop_quality_issues(df, report)
    assert "id_col" not in cleaned.columns


def test_summary_string():
    df = _clean_df(500)
    report = check_dataset_quality(df, target_column="label")
    summary = report.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_report_passed_property():
    df = _clean_df(500)
    report = check_dataset_quality(df, target_column="label", min_rows=200)
    # A clean 500-row df should pass
    assert report.passed is True


def test_report_failures_property():
    df = _clean_df(50)  # too few rows
    report = check_dataset_quality(df, target_column="label", min_rows=200)
    assert len(report.failures) > 0
    for f in report.failures:
        assert f.status == CheckStatus.FAIL
