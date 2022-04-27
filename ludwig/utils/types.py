from typing import Union

import pandas as pd

try:
    import dask.dataframe as dd

    DataFrame = Union[pd.DataFrame, dd.DataFrame]
    Series = Union[pd.Series, dd.Series]
except ImportError:
    DataFrame = pd.DataFrame
    Series = pd.Series
