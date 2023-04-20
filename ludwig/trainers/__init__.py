# register trainers

import ludwig.trainers.trainer  # noqa: F401

try:
    import ludwig.trainers.trainer_lightgbm  # noqa: F401
except ImportError:
    pass


try:
    import ludwig.trainers.trainer_llm  # noqa: F401
except ImportError:
    pass
