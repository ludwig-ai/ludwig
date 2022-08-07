import argparse
import importlib
import os
import pkgutil
import shutil
from typing import List

import yaml

from ludwig.datasets.registry import dataset_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig


def _import_submodules():
    from ludwig import datasets

    for _, name, _ in pkgutil.walk_packages(datasets.__path__):
        if name not in {"archives", "dataset", "kaggle"}:
            full_name = datasets.__name__ + "." + name
            importlib.import_module(full_name)


def _import_dataset_configs():
    """Generates dataset instances from config files.

    Must be called after _import_submodules for those configs which require a custom implementation.
    """
    from ludwig.datasets import configs
    from ludwig.datasets.dataset import Dataset, DatasetConfig

    config_files = [f for f in importlib.resources.contents(configs) if f.endswith(".yaml")]
    for config_file in config_files:
        config_path = os.path.join(os.path.dirname(configs.__file__), config_file)
        with open(config_path) as f:
            dataset_config = DatasetConfig(**yaml.safe_load(f))
            dataset_registry[dataset_config.name] = Dataset(dataset_config)


_import_submodules()
_import_dataset_configs()

# TODO: generate datasets from configs


def list_datasets() -> List[str]:
    return list(dataset_registry.keys())


def describe_dataset(dataset: str) -> str:
    return dataset_registry[dataset].__doc__


def download_dataset(dataset: str, output_dir: str = "."):
    dataset_obj = dataset_registry[dataset]()
    if not dataset_obj.is_processed():
        dataset_obj.process()
    processed_path = dataset_obj.processed_dataset_path
    _copytree(processed_path, output_dir)


def _copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


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
