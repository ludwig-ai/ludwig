from datetime import datetime, timedelta
from itertools import combinations
from random import randrange
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ludwig.data.dataframe.pandas import PandasEngine
from ludwig.data.split import get_splitter

try:
    from ludwig.data.dataframe.dask import DaskEngine
except ImportError:
    DaskEngine = Mock


def test_make_divisions_ensure_minimum_rows():
    from ludwig.data.split import _make_divisions_ensure_minimum_rows

    # Constraints are satisfied, the function should make no change to divisions.
    divisions = _make_divisions_ensure_minimum_rows((70, 80), 100, min_val_rows=3, min_test_rows=3)
    assert divisions[0] == 70
    assert divisions[1] == 80
    # Constraints are satisfied, the function should make no change to divisions.
    divisions = _make_divisions_ensure_minimum_rows((20, 22), 25, min_val_rows=0, min_test_rows=0)
    assert divisions[0] == 20
    assert divisions[1] == 22
    # The number of rows in validation set is too small.
    divisions = _make_divisions_ensure_minimum_rows((17, 19), 25, min_val_rows=3, min_test_rows=3)
    assert divisions[0] == 16
    assert divisions[1] == 19
    # The number of rows in validation and test sets are both too small.
    divisions = _make_divisions_ensure_minimum_rows((20, 22), 25, min_val_rows=3, min_test_rows=3)
    assert divisions[0] == 19
    assert divisions[1] == 22


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_random_split(df_engine, ray_cluster_2cpu):
    nrows = 100
    npartitions = 10

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])
    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    probs = (0.7, 0.1, 0.2)
    split_params = {
        "type": "random",
        "probabilities": probs,
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend, random_seed=42)

    assert len(splits) == 3
    for split, p in zip(splits, probs):
        if isinstance(df_engine, DaskEngine):
            # Dask splitting is not exact, so apply soft constraint here
            assert np.isclose(len(split), int(nrows * p), atol=5)
        else:
            assert len(split) == int(nrows * p)

    # Test determinism
    def compute(dfs):
        return [df.compute() if isinstance(backend.df_engine, DaskEngine) else df for df in dfs]

    splits = compute(splits)
    splits2 = compute(splitter.split(df, backend, random_seed=7))
    for s1, s2 in zip(splits, splits2):
        assert not s1.equals(s2)

    splits3 = compute(splitter.split(df, backend, random_seed=42))
    for s1, s3 in zip(splits, splits3):
        assert s1.equals(s3)


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_random_split_zero_probability_for_test_produces_no_zombie(df_engine, ray_cluster_2cpu):
    nrows = 102
    npartitions = 10

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])
    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    probs = (0.7, 0.3, 0.0)
    split_params = {
        "type": "random",
        "probabilities": probs,
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend, random_seed=42)

    assert len(splits[-1]) == 0


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_fixed_split(df_engine, ray_cluster_2cpu):
    nrows = 100
    npartitions = 10
    thresholds = [60, 80, 100]

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])

    def get_split(v):
        if v < thresholds[0]:
            return 0
        if thresholds[0] <= v < thresholds[1]:
            return 1
        return 2

    df["split_col"] = df["C"].map(get_split).astype(np.int8)

    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    split_params = {
        "type": "fixed",
        "column": "split_col",
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend)

    assert len(splits) == 3

    last_t = 0
    for split, t in zip(splits, thresholds):
        if isinstance(df_engine, DaskEngine):
            split = split.compute()

        assert np.all(split["C"] < t)
        assert np.all(split["C"] >= last_t)
        last_t = t


@pytest.mark.parametrize(
    ("df_engine", "nrows", "atol"),
    [
        pytest.param(PandasEngine(), 100, 1, id="pandas"),
        # Splitting with a distributed engine becomes more accurate with more rows.
        pytest.param(DaskEngine(_use_ray=False), 10000, 10, id="dask", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize(
    "class_probs",
    [
        pytest.param(np.array([0.33, 0.33, 0.34]), id="balanced"),
        pytest.param(np.array([0.6, 0.2, 0.2]), id="imbalanced"),
    ],
)
def test_stratify_split(df_engine, nrows, atol, class_probs, ray_cluster_2cpu):
    npartitions = 10
    thresholds = np.cumsum((class_probs * nrows).astype(int))

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])

    def get_category(v):
        if v < thresholds[0]:
            return 0
        if thresholds[0] <= v < thresholds[1]:
            return 1
        return 2

    df["category"] = df.index.map(get_category).astype(np.int8)

    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    probs = (0.7, 0.1, 0.2)
    split_params = {
        "type": "stratify",
        "column": "category",
        "probabilities": probs,
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend, random_seed=42)
    assert len(splits) == 3

    ratios = class_probs * nrows
    for split, p in zip(splits, probs):
        if isinstance(df_engine, DaskEngine):
            split = split.compute()
        for idx, r in enumerate(ratios):
            actual = np.sum(split["category"] == idx)
            expected = int(r * p)
            assert np.isclose(actual, expected, atol=atol)

    # Test determinism
    splits2 = splitter.split(df, backend, random_seed=7)
    for s1, s2 in zip(splits, splits2):
        if isinstance(df_engine, DaskEngine):
            s1 = s1.compute()
            s2 = s2.compute()
        assert not s1.equals(s2)

    splits3 = splitter.split(df, backend, random_seed=42)
    for s1, s3 in zip(splits, splits3):
        if isinstance(df_engine, DaskEngine):
            s1 = s1.compute()
            s3 = s3.compute()
        assert s1.equals(s3)


@pytest.mark.parametrize(
    ("df_engine", "atol"),
    [
        pytest.param(PandasEngine(), 1, id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), 10, id="dask", marks=pytest.mark.distributed),
    ],
)
def test_single_occurrence_stratified_split(df_engine, atol, ray_cluster_2cpu):
    nrows = 1000
    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 2)), columns=["A", "B"])
    # create 4 classes, where two of them each occurs once in the dataframe.
    df["category"] = (nrows // 2 - 1) * [0, 1] + [2, 3]

    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=10)

    probs = (0.7, 0.1, 0.2)
    split_params = {
        "type": "stratify",
        "column": "category",
        "probabilities": probs,
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend, random_seed=42)
    assert len(splits) == 3

    ratios = np.array([0.499, 0.499, 0.001, 0.001]) * nrows
    for split, p in zip(splits, probs):
        if isinstance(df_engine, DaskEngine):
            split = split.compute()
        for idx, r in enumerate(ratios):
            actual = np.sum(split["category"] == idx)
            expected = int(r * p)
            assert np.isclose(actual, expected, atol=atol)


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_datetime_split(df_engine, ray_cluster_2cpu):
    nrows = 100
    npartitions = 10

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])

    def random_date(*args, **kwargs):
        start = datetime.strptime("1/1/1990 1:30 PM", "%m/%d/%Y %I:%M %p")
        end = datetime.strptime("1/1/2022 4:50 AM", "%m/%d/%Y %I:%M %p")
        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = randrange(int_delta)
        return str(start + timedelta(seconds=random_second))

    df["date_col"] = df["C"].map(random_date)

    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    probs = (0.7, 0.1, 0.2)
    split_params = {
        "type": "datetime",
        "column": "date_col",
        "probabilities": probs,
    }
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend)

    assert len(splits) == 3

    min_datestr = "1990-01-01 00:00:00"
    for split, p in zip(splits, probs):
        if isinstance(df_engine, DaskEngine):
            # Dask splitting is not exact, so apply soft constraint here
            split = split.compute()
            assert np.isclose(len(split), int(nrows * p), atol=15)
        else:
            assert len(split) == int(nrows * p)

        assert np.all(split["date_col"] > min_datestr)
        min_datestr = split["date_col"].max()


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_hash_split(df_engine, ray_cluster_2cpu):
    nrows = 100
    npartitions = 10

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])
    df["id"] = np.arange(0, 100)

    if isinstance(df_engine, DaskEngine):
        df = df_engine.df_lib.from_pandas(df, npartitions=npartitions)

    probabilities = [0.8, 0.1, 0.1]
    split_params = {"type": "hash", "column": "id", "probabilities": probabilities}
    splitter = get_splitter(**split_params)

    backend = Mock()
    backend.df_engine = df_engine
    splits = splitter.split(df, backend)
    assert len(splits) == 3
    if isinstance(df_engine, DaskEngine):
        splits = [split.compute() for split in splits]

    # IDs should not overlap between splits
    assert all([set(split1["id"]).isdisjoint(set(split2["id"])) for split1, split2 in combinations(splits, 2)])

    for split, p in zip(splits, probabilities):
        # Should be approximately the same size as the desired proportion
        assert nrows * p - 5 <= len(split["id"]) <= nrows * p + 5

    # Need to ensure deterministic splitting even as we append data
    df2 = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])
    df2["id"] = np.arange(100, 200)

    nrows *= 2
    df = df.append(df2)

    splits2 = splitter.split(df, backend)
    assert len(splits2) == 3
    if isinstance(df_engine, DaskEngine):
        splits2 = [split.compute() for split in splits2]

    # IDs should not overlap between splits
    assert all([set(split1["id"]).isdisjoint(set(split2["id"])) for split1, split2 in combinations(splits2, 2)])

    for split1, split2, p in zip(splits, splits2, probabilities):
        ids1 = set(split1["id"].values.tolist())
        ids2 = set(split2["id"].values.tolist())

        assert nrows * p - 10 <= len(ids2) <= nrows * p + 10

        # All elements from the first round of splitting are in the same split, even after appending
        # more rows
        assert ids1.issubset(ids2)
