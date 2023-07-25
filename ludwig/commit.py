import argparse
import logging
import sys
from typing import Optional

from ludwig.utils.commit_utils import HuggingFaceHub
from ludwig.utils.print_utils import get_logging_level_registry

logger = logging.getLogger(__name__)


def commit_cli(
    repo_id: str,
    model_path: str,
    repo_type: str = "model",
    private: bool = False,
    commit_message: str = "Upload trained [Ludwig](https://ludwig.ai/latest/) model weights",
    commit_description: Optional[str] = None,
    **kwargs,
) -> None:
    """Create an empty repo on the HuggingFace Hub and upload trained model artifacts to that repo.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        model_path (`str`):
            The path of the saved model. This is the top level directory where
            the models weights as well as other associated training artifacts
            are saved.
        private (`bool`, *optional*, defaults to `False`):
            Whether the model repo should be private.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model. Default is
            `None`.
        commit_message (`str`, *optional*):
            The summary / title / first line of the generated commit. Defaults to:
            `f"Upload {path_in_repo} with huggingface_hub"`
        commit_description (`str` *optional*):
            The description of the generated commit
    """

    hf = HuggingFaceHub()
    hf.login()
    hf.upload_to_hfhub(
        repo_id=repo_id,
        model_path=model_path,
        repo_type=repo_type,
        private=private,
        commit_message=commit_message,
        commit_description=commit_description,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script pushes a trained model to huggingface hub",
        prog="ludwig commit",
        usage="%(prog)s [options]",
    )

    # ---------------
    # Required parameters
    # ---------------
    parser.add_argument(
        "-r",
        "--repo_id",
        help="ID of the repo on HF. This will be created if it doesn't exist. " "Format: username/repo_name",
        required=True,
    )

    parser.add_argument("-m", "--model_path", help="model to push to huggingface", required=True)

    # ---------------
    # Optional parameters
    # ---------------
    parser.add_argument("-p", "--private", help="make the repo private", default=False, choices=[True, False])

    parser.add_argument(
        "-t", "--repo_type", help="type of repo", default="model", choices=["model", "space", "dataset"]
    )

    parser.add_argument(
        "-c",
        "--commit_message",
        help="The summary / title / first line of the generated commit.",
        default="Upload trained [Ludwig](https://ludwig.ai/latest/) model weights",
    )

    parser.add_argument("-d", "--commit_description", help="The description of the generated commit", default=None)

    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    args = parser.parse_args(sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.commit")

    commit_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
