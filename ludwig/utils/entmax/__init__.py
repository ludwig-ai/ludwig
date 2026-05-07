__version__ = "1.1.dev0"

from ludwig.utils.entmax.activations import Entmax15, entmax15, Sparsemax, sparsemax
from ludwig.utils.entmax.losses import (
    entmax15_loss,
    Entmax15Loss,
    entmax_bisect_loss,
    EntmaxBisectLoss,
    sparsemax_bisect_loss,
    sparsemax_loss,
    SparsemaxBisectLoss,
    SparsemaxLoss,
)
from ludwig.utils.entmax.root_finding import entmax_bisect, EntmaxBisect, sparsemax_bisect, SparsemaxBisect

__all__ = [
    "Entmax15",
    "Entmax15Loss",
    "EntmaxBisect",
    "EntmaxBisectLoss",
    "Sparsemax",
    "SparsemaxBisect",
    "SparsemaxBisectLoss",
    "SparsemaxLoss",
    "entmax15",
    "entmax15_loss",
    "entmax_bisect",
    "entmax_bisect_loss",
    "sparsemax",
    "sparsemax_bisect",
    "sparsemax_bisect_loss",
    "sparsemax_loss",
]
