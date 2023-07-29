import argparse
import logging
import sys
from typing import Optional

from ludwig.utils.print_utils import get_logging_level_registry
from ludwig.utils.upload_utils import HuggingFaceHub

logger = logging.getLogger(__name__)


def get_upload_registry():
    return {
        "hf_hub": HuggingFaceHub,
    }


def upload_cli(
    service: str,
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
        service (`str`):
            Name of the hosted model service to push the trained artifacts to.
            Currently, this only supports `hf_hub`.
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
    model_service = get_upload_registry().get(service, "hf_hub")
    hub = model_service()
    hub.login()
    hub.upload(
        repo_id=repo_id,
        model_path=model_path,
        repo_type=repo_type,
        private=private,
        commit_message=commit_message,
        commit_description=commit_description,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script pushes a trained model to a hosted model repository service",
        prog="ludwig upload",
        usage="%(prog)s [options]",
    )

    # ---------------
    # Required parameters
    # ---------------
    parser.add_argument(
        "service",
        help="Name of the model repository service.",
        default="hf_hub",
        choices=["hf_hub"],
    )

    parser.add_argument(
        "-r",
        "--repo_id",
        help="Name of the repo. This will be created if it doesn't exist. Format: username/repo_name",
        required=True,
    )

    parser.add_argument("-m", "--model_path", help="Path of the trained model on disk", required=True)

    # ---------------
    # Optional parameters
    # ---------------
    parser.add_argument("-p", "--private", help="Make the repo private", default=False, choices=[True, False])

    parser.add_argument(
        "-t", "--repo_type", help="Type of repo", default="model", choices=["model", "space", "dataset"]
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
        help="The level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    args = parser.parse_args(sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.upload")

    upload_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
