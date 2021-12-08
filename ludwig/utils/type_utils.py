from typing import Union

import pandas as pd

try:
    import dask.dataframe as dd

    Column = Union[str, pd.Series, dd.Series]
except ImportError:
    Column = Union[str, pd.Series]
