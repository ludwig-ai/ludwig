# register trainers

import ludwig.trainers.trainer

try:
    import ludwig.trainers.trainer_dpo
    import ludwig.trainers.trainer_llm  # noqa: F401
except ImportError:
    pass
