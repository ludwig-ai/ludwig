from typing import List, Tuple, Union

import pandas as pd
import torch

try:
    import dask.dataframe as dd

    DataFrame = Union[pd.DataFrame, dd.DataFrame]
    Series = Union[pd.Series, dd.Series]
except ImportError:
    DataFrame = pd.DataFrame
    Series = pd.Series

TorchscriptPreprocessingInput = Union[List[str], List[torch.Tensor], List[Tuple[torch.Tensor, int]], torch.Tensor]
