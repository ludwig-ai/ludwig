__version__ = "1.1.dev0"

from ludwig.utils.entmax.activations import (Entmax15, Sparsemax, entmax15,
                                             sparsemax)
from ludwig.utils.entmax.losses import (Entmax15Loss, EntmaxBisectLoss,
                                        SparsemaxBisectLoss, SparsemaxLoss,
                                        entmax15_loss, entmax_bisect_loss,
                                        sparsemax_bisect_loss, sparsemax_loss)
from ludwig.utils.entmax.root_finding import (EntmaxBisect, SparsemaxBisect,
                                              entmax_bisect, sparsemax_bisect)

__all__ = [
    "entmax15",
    "Entmax15",
    "sparsemax",
    "Sparsemax",
    "entmax15_loss",
    "Entmax15Loss",
    "entmax_bisect_loss",
    "EntmaxBisectLoss",
    "sparsemax_bisect_loss",
    "sparsemax_loss",
    "SparsemaxBisectLoss",
    "SparsemaxLoss",
    "entmax_bisect",
    "EntmaxBisect",
    "sparsemax_bisect",
    "SparsemaxBisect",
]
