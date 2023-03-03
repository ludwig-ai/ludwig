import argparse
import logging
import tempfile

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import INPUT_FEATURES, OUTPUT_FEATURES, TRAINER
from ludwig.data.dataset_synthesizer import build_synthetic_dataset_df
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

NUM_EXAMPLES = 100


@DeveloperAPI
def check_install(logging_level: int = logging.INFO, **kwargs):
    config = {
        INPUT_FEATURES: [
            {"name": "in1", "type": "text"},
            {"name": "in2", "type": "category"},
            {"name": "in3", "type": "number"},
        ],
        OUTPUT_FEATURES: [{"name": "out1", "type": "binary"}],
        TRAINER: {"epochs": 2, "batch_size": 8},
    }

    try:
        df = build_synthetic_dataset_df(NUM_EXAMPLES, config)
        model = LudwigModel(config, logging_level=logging_level)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.train(dataset=df, output_directory=tmpdir)
    except Exception:
        print("=== CHECK INSTALL COMPLETE... FAILURE ===")
        raise

    print("=== CHECK INSTALL COMPLETE... SUCCESS ===")


@DeveloperAPI
def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This command checks Ludwig installation on a synthetic dataset.",
        prog="ludwig check_install",
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-l",
        "--logging_level",
        default="warning",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    args = parser.parse_args(sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.check")

    print_ludwig("Check Install", LUDWIG_VERSION)
    check_install(**vars(args))
