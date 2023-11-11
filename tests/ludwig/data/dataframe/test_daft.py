import daft
import numpy as np
import pytest

from ludwig.data.dataframe.daft import DaftEngine, LudwigDaftDataframe, LudwigDaftSeries


@pytest.fixture(scope="function")
def df() -> LudwigDaftDataframe:
    data = {
        "a": [i for i in range(10)],
        "b": ["a" * i for i in range(10)],
        "c": [np.zeros((i, i)) for i in range(1, 11)],
    }
    return LudwigDaftDataframe(daft.from_pydict(data))


@pytest.fixture(scope="function", params=[1, 2])
def engine(request) -> DaftEngine:
    parallelism = request.param
    return DaftEngine(parallelism=parallelism)


def test_df_like(df: LudwigDaftDataframe, engine: DaftEngine):
    s1 = LudwigDaftSeries(df["a"].expr * 2)
    s2 = LudwigDaftSeries(df["b"].expr + "_suffix")
    df = engine.df_like(df, {"foo": s1, "bar": s2})
    pd_df = engine.compute(df)

    assert list(pd_df.columns) == ["a", "b", "c", "foo", "bar"]
    np.testing.assert_equal(np.array(pd_df["foo"]), np.array(pd_df["a"] * 2))
    np.testing.assert_equal(np.array(pd_df["bar"]), np.array([item + "_suffix" for item in pd_df["b"]]))
