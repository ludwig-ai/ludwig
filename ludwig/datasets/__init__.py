import argparse
import importlib
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ludwig.api_annotations import DeveloperAPI, PublicAPI
from ludwig.backend.base import Backend
from ludwig.constants import AUDIO, BINARY, CATEGORY, IMAGE, NUMBER, TEXT, TYPE
from ludwig.data.cache.types import CacheableDataframe
from ludwig.datasets import configs
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

# PublicAPI
from ludwig.datasets.utils import model_configs_for_dataset  # noqa
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig
from ludwig.utils.types import DataFrame

URI_PREFIX = "ludwig://"


def _load_dataset_config(config_filename: str):
    """Loads a dataset config."""
    config_path = os.path.join(os.path.dirname(configs.__file__), config_filename)
    with open(config_path) as f:
        return DatasetConfig.from_dict(yaml.safe_load(f))


@lru_cache(maxsize=1)
def _get_dataset_configs() -> Dict[str, DatasetConfig]:
    """Returns all dataset configs indexed by name."""
    import importlib.resources

    config_files = [f for f in importlib.resources.contents(configs) if f.endswith(".yaml")]
    config_objects = [_load_dataset_config(f) for f in config_files]
    return {c.name: c for c in config_objects}


def _get_dataset_config(dataset_name) -> DatasetConfig:
    """Get the config for a dataset."""
    configs = _get_dataset_configs()
    if dataset_name not in configs:
        raise AttributeError(f"No config found for dataset {dataset_name}")
    return configs[dataset_name]


@PublicAPI
def get_dataset(dataset_name, cache_dir=None) -> DatasetLoader:
    """Gets an instance of the dataset loader for a dataset."""
    config = _get_dataset_config(dataset_name)
    class_name = config.loader.split(".")[-1]
    module_name = "." + ".".join(config.loader.split(".")[:-1])
    loader_module = importlib.import_module(module_name, package="ludwig.datasets.loaders")
    loader_cls = getattr(loader_module, class_name)
    if cache_dir:
        return loader_cls(config, cache_dir=cache_dir)
    return loader_cls(config)


@DeveloperAPI
def load_dataset_uris(
    dataset: Optional[Union[str, DataFrame]],
    training_set: Optional[Union[str, DataFrame]],
    validation_set: Optional[Union[str, DataFrame]],
    test_set: Optional[Union[str, DataFrame]],
    backend: Backend,
) -> Tuple[
    Optional[CacheableDataframe],
    Optional[CacheableDataframe],
    Optional[CacheableDataframe],
    Optional[CacheableDataframe],
]:
    """Loads and returns any Ludwig dataset URIs as CacheableDataframes.

    Returns the input unmodified for any non-Ludwig datasets.
    """
    dataset_out = dataset
    training_set_out = training_set
    validation_set_out = validation_set
    test_set_out = test_set

    # Check that any of the datasets begin with the `ludwig://` prefix denoting a Ludwig dataset URI
    if dataset is not None:
        if isinstance(dataset, str) and dataset.startswith(URI_PREFIX):
            dataset_out = _load_cacheable_dataset(dataset, backend)
    elif training_set is not None:
        train_df = test_df = val_df = None
        training_set_checksum = None
        if isinstance(training_set, str) and training_set.startswith(URI_PREFIX):
            # For the training set, we only want to use the TRAINING split of the dataset
            dataset_name = training_set[len(URI_PREFIX) :]
            loader = get_dataset(dataset_name)
            train_df, test_df, val_df = loader.load(split=True)
            training_set_checksum = str(loader.get_mtime())
            train_df = backend.df_engine.from_pandas(train_df)
            training_set_out = CacheableDataframe(df=train_df, name=training_set, checksum=training_set_checksum)

        if isinstance(validation_set, str) and validation_set.startswith(URI_PREFIX):
            if validation_set == training_set:
                # Reuse the loaded DF from the training split
                val_df = backend.df_engine.from_pandas(val_df)
                validation_set_out = CacheableDataframe(df=val_df, name=validation_set, checksum=training_set_checksum)
            else:
                validation_set_out = _load_cacheable_dataset(validation_set, backend)

        if isinstance(test_set, str) and test_set.startswith(URI_PREFIX):
            if test_set == training_set:
                # Reuse the loaded DF from the training split
                test_df = backend.df_engine.from_pandas(test_df)
                test_set_out = CacheableDataframe(df=test_df, name=test_set, checksum=training_set_checksum)
            else:
                test_set_out = _load_cacheable_dataset(test_set, backend)

    return dataset_out, training_set_out, validation_set_out, test_set_out


def _load_cacheable_dataset(dataset: str, backend: Backend) -> CacheableDataframe:
    dataset_name = dataset[len(URI_PREFIX) :]
    loader = get_dataset(dataset_name)
    df = loader.load(split=False)
    df = backend.df_engine.from_pandas(df)
    return CacheableDataframe(df=df, name=dataset, checksum=str(loader.get_mtime()))


@PublicAPI
def list_datasets() -> List[str]:
    """Returns a list of the names of all available datasets."""
    return sorted(_get_dataset_configs().keys())


@PublicAPI
def get_datasets_output_features(
    dataset: str = None, include_competitions: bool = True, include_data_modalities: bool = False
) -> dict:
    """Returns a dictionary with the output features for each dataset. Optionally, you can pass a dataset name
    which will then cause the function to return a dictionary with the output features for that dataset.

    :param dataset: (str) name of the dataset
    :param include_competitions: (bool) whether to include the output features from kaggle competition datasets
    :param include_data_modalities: (bool) whether to include the data modalities associated with the prediction task
    :return: (dict) dictionary with the output features for each dataset or a dictionary with the output features for
                    the specified dataset
    """
    ordered_configs = OrderedDict(sorted(_get_dataset_configs().items()))
    competition_datasets = []

    for name, config in ordered_configs.items():
        if not include_competitions and config.kaggle_competition:
            competition_datasets.append(name)
            continue

        ordered_configs[name] = {"name": config.name, "output_features": config.output_features}

        if include_data_modalities:
            column_types = {column[TYPE] for column in config.columns}

            data_modalities = set()
            if NUMBER in column_types or CATEGORY in column_types or BINARY in column_types:
                data_modalities.add("Tabular")
            if TEXT in column_types:
                data_modalities.add("Text")
            if IMAGE in column_types:
                data_modalities.add("Image")
            if AUDIO in column_types:
                data_modalities.add("Audio")

            ordered_configs[name]["data_modalities"] = data_modalities

    if dataset:
        return ordered_configs[dataset]

    if not include_competitions:
        for competition in competition_datasets:
            del ordered_configs[competition]

    return ordered_configs


@PublicAPI
def describe_dataset(dataset_name: str) -> str:
    """Returns the description of the dataset."""
    return _get_dataset_configs()[dataset_name].description


@PublicAPI
def download_dataset(dataset_name: str, output_dir: str = "."):
    """Downloads the dataset to the specified directory."""
    output_dir = os.path.expanduser(os.path.normpath(output_dir))
    dataset = get_dataset(dataset_name)
    dataset.export(output_dir)


@DeveloperAPI
def get_buffer(dataset_name: str, kaggle_username: str = None, kaggle_key: str = None) -> BytesIO:
    """Returns a byte buffer for the specified dataset."""
    try:
        dataset = get_dataset(dataset_name).load(kaggle_username=kaggle_username, kaggle_key=kaggle_key)
        buffer = BytesIO(dataset.to_parquet())
        return buffer
    except Exception as e:
        logging.error(logging.ERROR, f"Failed to upload dataset {dataset_name}: {e}")


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This command downloads and lists Ludwig-ready datasets.",
        prog="ludwig datasets",
        usage="%(prog)s [options]",
    )
    sub_parsers = parser.add_subparsers(dest="command", help="download and list datasets")

    parser_download = sub_parsers.add_parser("download", help="download a dataset")
    parser_download.add_argument("dataset", help="dataset to download")
    parser_download.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="output directory to download into",
        required=False,
    )

    sub_parsers.add_parser("list", help="list datasets")

    parser_describe = sub_parsers.add_parser("describe", help="describe datasets")
    parser_describe.add_argument("dataset", help="dataset to describe")

    args = parser.parse_args(sys_argv)
    print_ludwig(f"Datasets {args.command}", LUDWIG_VERSION)

    if args.command == "list":
        datasets = list_datasets()
        for ds in datasets:
            print(ds)
    elif args.command == "describe":
        print(describe_dataset(args.dataset))
    elif args.command == "download":
        download_dataset(args.dataset, args.output_dir)
    else:
        raise ValueError(f"Unrecognized command: {args.command}")


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ allows us to return an instance of a class.  For example:

         from ludwig.datasets import titanic

    returns an instance of DatasetLoader configured to load titanic.

    If you want to download a dataset in a non-default ludwig cache directory, there are two options:
        1. set the LUDWIG_CACHE environment variable to your desired path before importing the dataset
        2. Use ludwig.datasets.get_dataset(dataset_name, cache_dir=<CACHE_DIR>)
    """
    public_methods = {
        "list_datasets",
        "describe_dataset",
        "download_dataset",
        "cli",
        "get_dataset",
        "model_configs_for_dataset",
    }
    if name in public_methods:
        return globals()[name]
    return get_dataset(name)
