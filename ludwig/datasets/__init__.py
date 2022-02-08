import argparse
import importlib
import pkgutil
from typing import List

from ludwig import datasets
from ludwig.datasets.registry import dataset_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig


def _import_submodules():
    for _, name, _ in pkgutil.walk_packages(datasets.__path__):
        full_name = datasets.__name__ + "." + name
        importlib.import_module(full_name)


_import_submodules()


def list_datasets() -> List[str]:
    return list(dataset_registry.keys())


def download_dataset(dataset: str, output_dir: str = "."):
    import importlib

    importlib.import_module(f"ludwig.datasets.{dataset}")


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

    args = parser.parse_args(sys_argv)
    print_ludwig(f"Datasets {args.command}", LUDWIG_VERSION)

    if args.command == "list":
        datasets = list_datasets()
        for ds in datasets:
            print(ds)
    elif args.command == "download":
        download_dataset(**vars(args))
    else:
        raise ValueError(f"Unrecognized command: {args.command}")
