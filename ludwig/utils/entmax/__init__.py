__version__ = "1.1.dev0"

from ludwig.utils.entmax.activations import sparsemax, entmax15, Sparsemax, Entmax15
from ludwig.utils.entmax.root_finding import (
    sparsemax_bisect,
    entmax_bisect,
    SparsemaxBisect,
    EntmaxBisect,
)
from ludwig.utils.entmax.losses import (
    sparsemax_loss,
    entmax15_loss,
    sparsemax_bisect_loss,
    entmax_bisect_loss,
    SparsemaxLoss,
    SparsemaxBisectLoss,
    Entmax15Loss,
    EntmaxBisectLoss,
)
