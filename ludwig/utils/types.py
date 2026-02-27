from typing import Union

import pandas as pd
import torch

try:
    import dask.dataframe as dd

    DataFrame = Union[pd.DataFrame, dd.DataFrame]
    Series = Union[pd.Series, dd.Series]
except ImportError:
    DataFrame = pd.DataFrame
    Series = pd.Series

# torchaudio.load returns the audio tensor and the sampling rate as a tuple.
TorchAudioTuple = tuple[torch.Tensor, int]
TorchscriptPreprocessingInput = Union[list[str], list[torch.Tensor], list[TorchAudioTuple], torch.Tensor]
TorchDevice = Union[str, torch.device]
