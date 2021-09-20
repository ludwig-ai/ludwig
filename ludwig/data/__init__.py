import sys
from typing import Union

try:
    import dask.dataframe as dd
except ImportError:
    pass
import pandas as pd


if 'dask.dataframe' in sys.modules:
    Column: Union[str, pd.Series, dd.Series]
else:
    Column: Union[str, pd.Series]
