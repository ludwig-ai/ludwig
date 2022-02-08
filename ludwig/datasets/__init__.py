import argparse

import ludwig.datasets
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import print_ludwig


def download(dataset: str, output_dir: str = "."):
    pass


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
        import pkgutil

        datasets = [ds.name for ds in pkgutil.iter_modules(ludwig.datasets.__path__) if ds.name != "mixins"]
        for ds in datasets:
            print(ds)
    elif args.command == "download":
        download(**vars(args))
    else:
        raise ValueError(f"Unrecognized command: {args.command}")
