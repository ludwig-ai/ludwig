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
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ludwig.api_annotations import DeveloperAPI

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@DeveloperAPI
@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: dict = field(default_factory=dict)


@DeveloperAPI
@dataclass
class DatasetQualityReport:
    dataset_name: str
    n_rows: int
    n_cols: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        """True if no FAIL checks."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def summary(self) -> str:
        """One-line summary string."""
        n_fail = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
        n_warn = sum(1 for c in self.checks if c.status == CheckStatus.WARN)
        n_pass = len(self.checks) - n_fail - n_warn
        overall = "PASS" if n_fail == 0 else "FAIL"
        return (
            f"[{overall}] Dataset '{self.dataset_name}' "
            f"({self.n_rows} rows, {self.n_cols} cols): "
            f"{n_pass} passed, {n_warn} warnings, {n_fail} failures"
        )


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------


def _check_minimum_size(df: pd.DataFrame, min_rows: int) -> CheckResult:
    n = len(df)
    if n < min_rows:
        return CheckResult(
            name="minimum_size",
            status=CheckStatus.FAIL,
            message=f"Dataset has only {n} rows; at least {min_rows} are required.",
            details={"n_rows": n, "min_rows": min_rows},
        )
    return CheckResult(
        name="minimum_size",
        status=CheckStatus.PASS,
        message=f"Dataset has {n} rows (minimum {min_rows}).",
        details={"n_rows": n, "min_rows": min_rows},
    )


def _check_minimum_features(df: pd.DataFrame, target_column: str | None, min_features: int) -> CheckResult:
    feature_cols = [c for c in df.columns if c != target_column]
    n = len(feature_cols)
    if n < min_features:
        return CheckResult(
            name="minimum_features",
            status=CheckStatus.FAIL,
            message=(f"Dataset has only {n} non-target feature column(s); at least {min_features} are required."),
            details={"n_features": n, "min_features": min_features},
        )
    return CheckResult(
        name="minimum_features",
        status=CheckStatus.PASS,
        message=f"Dataset has {n} non-target feature column(s) (minimum {min_features}).",
        details={"n_features": n, "min_features": min_features},
    )


def _check_missing_values(df: pd.DataFrame, max_missing_pct: float) -> CheckResult:
    total_cells = df.size
    if total_cells == 0:
        missing_pct = 0.0
    else:
        missing_pct = df.isnull().sum().sum() / total_cells

    if missing_pct > max_missing_pct:
        return CheckResult(
            name="missing_values",
            status=CheckStatus.WARN,
            message=(f"Dataset has {missing_pct:.1%} missing values overall (threshold {max_missing_pct:.1%})."),
            details={"missing_pct": missing_pct, "max_missing_pct": max_missing_pct},
        )
    return CheckResult(
        name="missing_values",
        status=CheckStatus.PASS,
        message=f"Missing values: {missing_pct:.1%} (threshold {max_missing_pct:.1%}).",
        details={"missing_pct": missing_pct, "max_missing_pct": max_missing_pct},
    )


def _check_constant_columns(df: pd.DataFrame) -> CheckResult:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        return CheckResult(
            name="constant_columns",
            status=CheckStatus.WARN,
            message=f"Found {len(constant_cols)} constant column(s): {constant_cols}.",
            details={"constant_columns": constant_cols},
        )
    return CheckResult(
        name="constant_columns",
        status=CheckStatus.PASS,
        message="No constant columns detected.",
        details={"constant_columns": []},
    )


_MAX_CORR_COLS = 50


def _check_near_duplicate_columns(df: pd.DataFrame, threshold: float) -> CheckResult:
    """Finds pairs of numeric columns whose Pearson |r| exceeds *threshold*."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > _MAX_CORR_COLS:
        logger.warning(
            f"Skipping near-duplicate column check: {len(numeric_cols)} numeric columns exceeds "
            f"the {_MAX_CORR_COLS}-column cap to avoid O(n²) runtime."
        )
        return CheckResult(
            name="near_duplicate_columns",
            status=CheckStatus.WARN,
            message=f"Too many numeric columns ({len(numeric_cols)}) to check for near-duplicates efficiently.",
            details={"pairs": [], "threshold": threshold},
        )
    near_dup_pairs: list[tuple[str, str, float]] = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_a = numeric_cols[i]
            col_b = numeric_cols[j]
            try:
                r = df[[col_a, col_b]].dropna().corr().iloc[0, 1]
            except Exception:
                continue
            if pd.notna(r) and abs(r) > threshold:
                near_dup_pairs.append((col_a, col_b, float(r)))

    if near_dup_pairs:
        pair_strs = [f"({a}, {b}, r={r:.3f})" for a, b, r in near_dup_pairs]
        return CheckResult(
            name="near_duplicate_columns",
            status=CheckStatus.WARN,
            message=(f"Found {len(near_dup_pairs)} near-duplicate column pair(s) with |r| > {threshold}: {pair_strs}."),
            details={"pairs": near_dup_pairs, "threshold": threshold},
        )
    return CheckResult(
        name="near_duplicate_columns",
        status=CheckStatus.PASS,
        message=f"No near-duplicate column pairs found (threshold |r| > {threshold}).",
        details={"pairs": [], "threshold": threshold},
    )


def _check_target_leakage(df: pd.DataFrame, target_column: str, threshold: float = 0.99) -> CheckResult:
    """Detects features that are almost perfectly correlated with the target."""
    if target_column not in df.columns:
        return CheckResult(
            name="target_leakage",
            status=CheckStatus.WARN,
            message=f"Target column '{target_column}' not found in DataFrame; skipping leakage check.",
            details={},
        )

    target_series = df[target_column]
    if not pd.api.types.is_numeric_dtype(target_series):
        # Encode categorically for correlation purposes.
        target_encoded = target_series.astype("category").cat.codes.replace(-1, np.nan)
    else:
        target_encoded = target_series

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    leaking: list[tuple[str, float]] = []

    for col in feature_cols:
        try:
            r = df[[col]].join(target_encoded.rename("__target__")).dropna().corr().iloc[0, 1]
        except Exception:
            continue
        if pd.notna(r) and abs(r) > threshold:
            leaking.append((col, float(r)))

    if leaking:
        leaking_strs = [f"{c} (r={r:.3f})" for c, r in leaking]
        return CheckResult(
            name="target_leakage",
            status=CheckStatus.FAIL,
            message=(
                f"Possible target leakage: {len(leaking)} feature(s) are nearly perfectly "
                f"correlated with '{target_column}': {leaking_strs}."
            ),
            details={"leaking_columns": leaking, "threshold": threshold},
        )
    return CheckResult(
        name="target_leakage",
        status=CheckStatus.PASS,
        message=f"No target leakage detected (threshold |r| > {threshold}).",
        details={"leaking_columns": [], "threshold": threshold},
    )


def _check_id_columns(df: pd.DataFrame) -> CheckResult:
    """Identifies columns that look like identifiers."""
    id_cols: list[str] = []
    for col in df.columns:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue
        n_distinct = series.nunique()
        if n_distinct == n:
            id_cols.append(col)
            continue
        if pd.api.types.is_integer_dtype(series):
            unique_sorted = sorted(series.unique())
            if len(unique_sorted) >= 2:
                # Check sequential without allocating a full range list
                lo, hi = int(unique_sorted[0]), int(unique_sorted[-1])
                if hi - lo + 1 == len(unique_sorted):
                    id_cols.append(col)

    if id_cols:
        return CheckResult(
            name="id_columns",
            status=CheckStatus.WARN,
            message=f"Found {len(id_cols)} likely ID column(s): {id_cols}.",
            details={"id_columns": id_cols},
        )
    return CheckResult(
        name="id_columns",
        status=CheckStatus.PASS,
        message="No ID columns detected.",
        details={"id_columns": []},
    )


def _check_class_imbalance(df: pd.DataFrame, target_column: str) -> CheckResult:
    """Warns if the minority class represents less than 1% of total rows."""
    if target_column not in df.columns:
        return CheckResult(
            name="class_imbalance",
            status=CheckStatus.WARN,
            message=f"Target column '{target_column}' not found; skipping imbalance check.",
            details={},
        )

    target_series = df[target_column].dropna()
    if len(target_series) == 0:
        return CheckResult(
            name="class_imbalance",
            status=CheckStatus.WARN,
            message=f"Target column '{target_column}' has no non-null values; skipping imbalance check.",
            details={},
        )
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
        # Continuous target — not a classification problem.
        return CheckResult(
            name="class_imbalance",
            status=CheckStatus.PASS,
            message="Target appears continuous; class imbalance check skipped.",
            details={},
        )

    counts = target_series.value_counts()
    total = len(target_series)
    minority_count = int(counts.min())
    minority_class = counts.idxmin()
    minority_pct = minority_count / total if total > 0 else 0.0

    if minority_pct < 0.01:
        return CheckResult(
            name="class_imbalance",
            status=CheckStatus.WARN,
            message=(
                f"Class imbalance detected: minority class '{minority_class}' "
                f"has only {minority_count} sample(s) ({minority_pct:.2%} of rows)."
            ),
            details={
                "minority_class": str(minority_class),
                "minority_count": minority_count,
                "minority_pct": minority_pct,
            },
        )
    return CheckResult(
        name="class_imbalance",
        status=CheckStatus.PASS,
        message=f"Class balance is acceptable (minority class: {minority_pct:.2%}).",
        details={
            "minority_class": str(minority_class),
            "minority_count": minority_count,
            "minority_pct": minority_pct,
        },
    )


def _check_single_class(df: pd.DataFrame, target_column: str) -> CheckResult:
    """Fails if the target column has only one distinct value."""
    if target_column not in df.columns:
        return CheckResult(
            name="single_class",
            status=CheckStatus.WARN,
            message=f"Target column '{target_column}' not found; skipping single-class check.",
            details={},
        )

    n_distinct = df[target_column].nunique(dropna=True)
    if n_distinct <= 1:
        return CheckResult(
            name="single_class",
            status=CheckStatus.FAIL,
            message=(
                f"Target column '{target_column}' has only {n_distinct} distinct value(s). "
                "A model cannot be trained on a single-class target."
            ),
            details={"n_distinct": n_distinct},
        )
    return CheckResult(
        name="single_class",
        status=CheckStatus.PASS,
        message=f"Target column '{target_column}' has {n_distinct} distinct value(s).",
        details={"n_distinct": n_distinct},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@DeveloperAPI
def check_dataset_quality(
    df: pd.DataFrame,
    target_column: str | None = None,
    dataset_name: str = "unnamed",
    min_rows: int = 200,
    min_features: int = 2,
    max_missing_pct: float = 0.5,
    max_correlation_threshold: float = 0.99,
) -> DatasetQualityReport:
    """Runs all quality checks on a DataFrame and returns a report.

    Checks performed:

    1. **minimum_size**: at least *min_rows* rows → FAIL if not.
    2. **minimum_features**: at least *min_features* non-target columns → FAIL if not.
    3. **missing_values**: overall missing value % > *max_missing_pct* → WARN.
    4. **constant_columns**: columns with only 1 distinct value → WARN (lists them).
    5. **near_duplicate_columns**: column pairs with Pearson |r| > *max_correlation_threshold* → WARN.
    6. **target_leakage**: any non-target feature with |r| > 0.99 with target → FAIL
       (only when *target_column* is provided).
    7. **id_columns**: columns that look like IDs (all-unique values or sequential ints) → WARN.
    8. **class_imbalance**: if target is categorical and minority class < 1% of rows → WARN
       (only when *target_column* is provided).
    9. **single_class**: if target has only 1 distinct value → FAIL
       (only when *target_column* is provided).

    # Inputs
    :param df: (pd.DataFrame) the dataset to check.
    :param target_column: (str | None) name of the target column, if known.
    :param dataset_name: (str) a label used in the report summary.
    :param min_rows: (int) minimum acceptable number of rows.
    :param min_features: (int) minimum acceptable number of non-target features.
    :param max_missing_pct: (float) maximum acceptable fraction of missing cells.
    :param max_correlation_threshold: (float) |r| above which two numeric columns
        are considered near-duplicates (also used for target leakage detection).

    # Return
    :return: (DatasetQualityReport) report containing all check results.
    """
    checks: list[CheckResult] = []

    checks.append(_check_minimum_size(df, min_rows))
    checks.append(_check_minimum_features(df, target_column, min_features))
    checks.append(_check_missing_values(df, max_missing_pct))
    checks.append(_check_constant_columns(df))
    checks.append(_check_near_duplicate_columns(df, max_correlation_threshold))
    checks.append(_check_id_columns(df))

    if target_column is not None:
        checks.append(_check_target_leakage(df, target_column, threshold=max_correlation_threshold))
        checks.append(_check_class_imbalance(df, target_column))
        checks.append(_check_single_class(df, target_column))

    report = DatasetQualityReport(
        dataset_name=dataset_name,
        n_rows=len(df),
        n_cols=len(df.columns),
        checks=checks,
    )

    logger.info(report.summary())
    for failure in report.failures:
        logger.warning("Quality FAIL — %s: %s", failure.name, failure.message)
    for warning in report.warnings:
        logger.info("Quality WARN — %s: %s", warning.name, warning.message)

    return report


@DeveloperAPI
def drop_quality_issues(df: pd.DataFrame, report: DatasetQualityReport) -> pd.DataFrame:
    """Returns a cleaned DataFrame with constant columns and detected ID columns removed.

    Columns removed:

    - Constant columns (identified by the *constant_columns* check).
    - Likely ID columns (identified by the *id_columns* check).

    # Inputs
    :param df: (pd.DataFrame) the original dataset.
    :param report: (DatasetQualityReport) quality report produced by
        :func:`check_dataset_quality`.

    # Return
    :return: (pd.DataFrame) cleaned copy of *df* with offending columns dropped.
    """
    cols_to_drop: set[str] = set()

    for check in report.checks:
        if check.name == "constant_columns":
            cols_to_drop.update(check.details.get("constant_columns", []))
        elif check.name == "id_columns":
            cols_to_drop.update(check.details.get("id_columns", []))

    # Only drop columns that actually exist in the DataFrame.
    cols_to_drop = cols_to_drop.intersection(df.columns)

    if cols_to_drop:
        logger.info("Dropping %d column(s) due to quality issues: %s", len(cols_to_drop), sorted(cols_to_drop))

    return df.drop(columns=list(cols_to_drop))
