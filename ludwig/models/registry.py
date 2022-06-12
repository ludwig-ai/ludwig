import logging

from ludwig.constants import MODEL_ECD, MODEL_GBM
from ludwig.models.ecd import ECD

model_type_registry = {
    MODEL_ECD: ECD,
}

try:
    import lightgbm  # noqa: F401

    from ludwig.models.gbm import GBM

    model_type_registry[MODEL_GBM] = GBM
except ImportError:
    logging.warning(
        "Importing GBM requirements failed. Not loading GBM model type. "
        "If you want to use GBM, install Ludwig's 'tree' extra."
    )
