# register trainers
import logging

import ludwig.trainers.trainer  # noqa: F401

try:
    import lightgbm  # noqa: F401

    import ludwig.trainers.trainer_lightgbm  # noqa: F401
except ImportError:
    logging.warning(
        "Importing GBM requirements failed. Not loading LightGBM trainer. "
        "If you want to use LightGBM, install Ludwig's 'tree' extra."
    )
