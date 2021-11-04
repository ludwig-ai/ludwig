from typing import Union

import pandas as pd

try:
    import dask.dataframe as dd
    DataFrame = Union[pd.DataFrame, dd.DataFrame]
except ImportError:
    DataFrame = pd.DataFrame
