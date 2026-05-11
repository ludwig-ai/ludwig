"""Ludwig AutoML — automated config generation and training.

Public API
----------
auto_train(dataset, target, time_limit_s)
    End-to-end: run dataset quality checks, sample configs from the YAML-driven
    search space, train each, and return the best model.

create_auto_config(dataset, target)
    Return a single sampled Ludwig config dict without training.

train_with_config(dataset, config, output_dir)
    Train one Ludwig config and return the LudwigModel.

cli_init_config(argv)
    CLI entry-point for `ludwig init_config`.

Config generation pipeline
--------------------------
The search space is defined by YAML files in ``ludwig/automl/search_space/``.
``config_enumerator`` builds the full combination lattice, ``config_sampler``
draws a random subset, and ``config_validator`` rejects invalid combinations
before any training starts.  ``target_detection`` infers the task type and
target column when they are not explicitly specified.
"""

from ludwig.automl.automl import (
    auto_train,  # noqa: F401
    cli_init_config,  # noqa: F401
    create_auto_config,  # noqa: F401
    train_with_config,  # noqa: F401
)
