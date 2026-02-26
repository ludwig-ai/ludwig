import os
from functools import lru_cache

import yaml

from ludwig.api_annotations import PublicAPI
from ludwig.datasets import model_configs


@PublicAPI
def model_configs_for_dataset(dataset_name: str) -> dict[str, dict]:
    """Returns a dictionary of built-in model configs for the specified dataset.

    Maps config name to ludwig config dict.
    """
    return _get_model_configs(dataset_name)


@lru_cache(maxsize=3)
def _get_model_configs(dataset_name: str) -> dict[str, dict]:
    """Returns all model configs for the specified dataset.

    Model configs are named <dataset_name>_<config_name>.yaml
    """
    import importlib.resources

    config_filenames = [
        f.name
        for f in importlib.resources.files(model_configs).iterdir()
        if f.name.endswith(".yaml") and f.name.startswith(dataset_name)
    ]
    configs = {}
    for config_filename in config_filenames:
        basename = os.path.splitext(config_filename)[0]
        config_name = basename[len(dataset_name) + 1 :]
        configs[config_name] = _load_model_config(config_filename)
    return configs


def _load_model_config(model_config_filename: str):
    """Loads a model config."""
    model_config_path = os.path.join(os.path.dirname(model_configs.__file__), model_config_filename)
    with open(model_config_path) as f:
        return yaml.safe_load(f)
