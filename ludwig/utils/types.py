import pandas as pd
import torch

try:
    import dask.dataframe as dd

    DataFrame = pd.DataFrame | dd.DataFrame
    Series = pd.Series | dd.Series
except ImportError:
    DataFrame = pd.DataFrame
    Series = pd.Series

# torchaudio.load returns the audio tensor and the sampling rate as a tuple.
TorchAudioTuple = tuple[torch.Tensor, int]
PreprocessingInput = list[str] | list[torch.Tensor] | list[TorchAudioTuple] | torch.Tensor
TorchDevice = str | torch.device
