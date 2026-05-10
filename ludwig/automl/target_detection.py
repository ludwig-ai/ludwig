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
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ludwig.api_annotations import DeveloperAPI

logger = logging.getLogger(__name__)

# Column names that strongly suggest a target / label role.
_TARGET_NAME_HINTS = {
    "target",
    "label",
    "y",
    "output",
    "class",
    "outcome",
    "result",
    "Target",
    "Label",
    "Class",
    "Y",
}

# Maximum number of distinct values for a column to be considered MULTICLASS
# when the column has numeric dtype.
_MULTICLASS_MAX_DISTINCT = 20

# Fraction of total rows below which a numeric column with few distinct values
# is still treated as MULTICLASS rather than REGRESSION.
_MULTICLASS_FRACTION_CUTOFF = 0.05

# Confidence values assigned by each heuristic tier.
_CONF_NAME_HINT = 0.95
_CONF_LAST_COLUMN = 0.60
_CONF_LOW_CARDINALITY_LAST_3 = 0.50
_CONF_BALANCED_BINARY = 0.45
_CONF_FEWEST_DISTINCT = 0.30

# Minimum confidence required to return a result.
_CONF_MIN_ACCEPTABLE = 0.25

# Balance thresholds for the balanced-binary heuristic.
_BALANCE_LOW = 0.30
_BALANCE_HIGH = 0.70


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


@DeveloperAPI
@dataclass
class TargetDetectionResult:
    column: str
    confidence: float  # 0.0–1.0
    task_type: TaskType
    reason: str  # Human-readable explanation


def _is_id_column(series: pd.Series) -> bool:
    """Returns True if a column looks like an ID (all-unique or sequential ints)."""
    n = len(series.dropna())
    if n == 0:
        return False
    n_distinct = series.nunique(dropna=True)
    if n_distinct == n:
        # All-unique — strong ID signal.
        return True
    # Sequential integers: sorted unique values form a contiguous range.
    if pd.api.types.is_integer_dtype(series):
        unique_sorted = sorted(series.dropna().unique())
        if len(unique_sorted) >= 2:
            expected = list(range(int(unique_sorted[0]), int(unique_sorted[-1]) + 1))
            if unique_sorted == expected:
                return True
    return False


@DeveloperAPI
def infer_task_type(series: pd.Series) -> TaskType:
    """Infers task type from a pandas Series (the target column values).

    # Inputs
    :param series: (pd.Series) the candidate target column.

    # Return
    :return: (TaskType) the inferred task type.
    """
    n_distinct = series.nunique(dropna=True)
    n_rows = len(series.dropna())

    if n_distinct == 2:
        return TaskType.BINARY

    if n_distinct <= _MULTICLASS_MAX_DISTINCT or (n_rows > 0 and n_distinct / n_rows <= _MULTICLASS_FRACTION_CUTOFF):
        return TaskType.MULTICLASS

    if pd.api.types.is_numeric_dtype(series):
        return TaskType.REGRESSION

    return TaskType.MULTICLASS


def _score_column(col: str, series: pd.Series, df: pd.DataFrame) -> tuple[float, str]:
    """Returns (confidence, reason) for a single candidate column.

    Applies heuristics in priority order and returns as soon as a match is
    found.  Lower-priority heuristics are only evaluated when none of the
    higher-priority ones fire.
    """
    n_distinct = series.nunique(dropna=True)
    n_rows = len(df)
    col_index = list(df.columns).index(col)
    last_index = len(df.columns) - 1

    # Heuristic 1 — well-known target name.
    if col in _TARGET_NAME_HINTS:
        return _CONF_NAME_HINT, f"Column name '{col}' is a well-known target/label name"

    # Heuristic 2 — last column in the DataFrame.
    if col_index == last_index:
        return _CONF_LAST_COLUMN, f"Column '{col}' is the last column in the DataFrame"

    # Heuristic 3 — lowest-cardinality non-binary, non-constant among last 3 columns.
    last_3 = list(df.columns[max(0, last_index - 2) :])
    if col in last_3 and 2 < n_distinct < n_rows:
        min_distinct_in_last3 = (
            min(df[c].nunique(dropna=True) for c in last_3 if df[c].nunique(dropna=True) > 2)
            if any(df[c].nunique(dropna=True) > 2 for c in last_3)
            else None
        )
        if min_distinct_in_last3 is not None and n_distinct == min_distinct_in_last3:
            return (
                _CONF_LOW_CARDINALITY_LAST_3,
                f"Column '{col}' has the lowest cardinality ({n_distinct}) among the last 3 columns",
            )

    # Heuristic 4 — binary column with near-50/50 balance.
    if n_distinct == 2:
        counts = series.value_counts(normalize=True, dropna=True)
        minority_frac = counts.min()
        if _BALANCE_LOW <= minority_frac <= _BALANCE_HIGH:
            return (
                _CONF_BALANCED_BINARY,
                f"Column '{col}' is binary with a balanced split (minority={minority_frac:.2f})",
            )

    # Heuristic 5 — not an ID column, fewest distinct values.
    if not _is_id_column(series) and n_distinct > 1:
        return (
            _CONF_FEWEST_DISTINCT,
            f"Column '{col}' is a non-ID column with {n_distinct} distinct values",
        )

    return 0.0, f"Column '{col}' did not match any target heuristic"


@DeveloperAPI
def detect_all_target_candidates(df: pd.DataFrame) -> list[TargetDetectionResult]:
    """Returns all plausible target columns sorted by confidence descending.

    # Inputs
    :param df: (pd.DataFrame) the dataset to analyse.

    # Return
    :return: (list[TargetDetectionResult]) candidates sorted by confidence.
    """
    if df.empty or len(df.columns) == 0:
        return []

    results: list[TargetDetectionResult] = []
    for col in df.columns:
        series = df[col]
        confidence, reason = _score_column(col, series, df)
        if confidence > 0.0:
            task_type = infer_task_type(series)
            results.append(
                TargetDetectionResult(
                    column=col,
                    confidence=confidence,
                    task_type=task_type,
                    reason=reason,
                )
            )

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


@DeveloperAPI
def detect_target_column(df: pd.DataFrame) -> TargetDetectionResult:
    """Infers the most likely target column from a DataFrame.

    Heuristics (ordered by decreasing confidence):

    1. Column named exactly: target, label, y, output, class, outcome, result,
       Target, Label, Class, Y → confidence 0.95
    2. Last column in the DataFrame → confidence 0.6
    3. Lowest-cardinality non-binary, non-constant column among the last 3
       columns → confidence 0.5
    4. Binary column with ~50/50 split (balance between 0.3–0.7) → confidence 0.45
    5. Column excluded from ID candidates (not all-unique, not sequential int)
       with fewest distinct values → confidence 0.3

    Task type is inferred from the winning column:

    - 2 distinct values → BINARY
    - ≤20 distinct values (or ≤5% of rows) → MULTICLASS
    - all-numeric values → REGRESSION

    # Inputs
    :param df: (pd.DataFrame) the dataset to analyse.

    # Return
    :return: (TargetDetectionResult) the most likely target column.

    :raises ValueError: if no column scores above 0.25 confidence.
    """
    if df.empty or len(df.columns) == 0:
        raise ValueError("Cannot detect target column: the DataFrame is empty or has no columns.")

    candidates = detect_all_target_candidates(df)
    if not candidates:
        raise ValueError(
            "Cannot detect target column: no columns in the DataFrame matched any target heuristic. "
            "Please specify the target column explicitly."
        )

    best = candidates[0]
    if best.confidence < _CONF_MIN_ACCEPTABLE:
        raise ValueError(
            f"Cannot detect target column with sufficient confidence "
            f"(best candidate '{best.column}' scored {best.confidence:.2f} < {_CONF_MIN_ACCEPTABLE}). "
            "Please specify the target column explicitly."
        )

    logger.info(
        f"Auto-detected target column '{best.column}' "
        f"(task={best.task_type.value}, confidence={best.confidence:.2f}): {best.reason}"
    )
    return best
