from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ludwig.data.split import get_splitter

try:
    from ludwig.data.dataframe.dask import DaskEngine
    from ludwig.data.dataframe.pandas import PandasEngine
except ImportError:
    DaskEngine = Mock


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_random_split(df_engine):
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
    splits = splitter.split(df, backend)

    assert len(splits) == 3
    for split, p in zip(splits, probs):
        if isinstance(df_engine, DaskEngine):
            # Dask splitting is not exact, so put softer constraint here
            assert np.isclose(len(split), int(nrows * p), atol=5)
        else:
            assert len(split) == int(nrows * p)
