import logging
import os
import tempfile
from os import PathLike
from typing import Optional, Tuple, Type, Union

from transformers import AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.error_handling_utils import default_retry
from ludwig.utils.fs_utils import download, path_exists
from ludwig.utils.upload_utils import hf_hub_login

logger = logging.getLogger(__name__)


@default_retry()
def load_pretrained_hf_model_from_hub(
    model_class: type,
    pretrained_model_name_or_path: str | PathLike | None,
    **pretrained_kwargs,
) -> PreTrainedModel:
    """Download a HuggingFace model.

    Downloads a model from the HuggingFace zoo with retry on failure.
    Args:
        model_class: Class of the model to download.
        pretrained_model_name_or_path: Name of the model to download.
        pretrained_kwargs: Additional arguments to pass to the model constructor.
    Returns:
        The pretrained model object.
    """
    return model_class.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)


@default_retry()
def load_pretrained_hf_tokenizer(
    pretrained_model_name_or_path: str | PathLike | None, **pretrained_kwargs
) -> PreTrainedTokenizer:
    """Download a HuggingFace tokenizer.

    Args:
        pretrained_model_name_or_path: Name of the tokenizer to download.
        pretrained_kwargs: Additional arguments to pass to the tokenizer constructor.
    Returns:
        The pretrained tokenizer object.
    """
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)


def _load_pretrained_hf_model_from_dir(
    model_class: type,
    pretrained_model_name_or_path: str | PathLike | None,
    **pretrained_kwargs,
) -> PreTrainedModel:
    """Downloads a model to a local temporary directory, and Loads a pretrained HF model from a local directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        download(pretrained_model_name_or_path, tmpdir)
        return model_class.from_pretrained(tmpdir, **pretrained_kwargs)


@DeveloperAPI
def load_pretrained_hf_model_with_hub_fallback(
    model_class: type,
    pretrained_model_name_or_path: str | PathLike | None,
    **pretrained_kwargs,
) -> tuple[PreTrainedModel, bool]:
    """Returns the model and a boolean indicating whether the model was downloaded from the HuggingFace hub.

    If the `LUDWIG_PRETRAINED_MODELS_DIR` environment variable is set, we attempt to load the HF model from this
    directory, falling back to downloading from the HF hub if the model is not found, downloading fails, or if model
    initialization fails.

    `LUDWIG_PRETRAINED_MODELS_DIR` can be an s3 path. Weights are copied to a local temporary directory, and the model
    is loaded from there.

    The expected structure of the `LUDWIG_PRETRAINED_MODELS_DIR` directory is:
        {LUDWIG_PRETRAINED_MODELS_DIR}/{pretrained_model_name_or_path}/pytorch_model.bin
        {LUDWIG_PRETRAINED_MODELS_DIR}/{pretrained_model_name_or_path}/config.json

    For example, if `LUDWIG_PRETRAINED_MODELS_DIR` is set to `s3://my-bucket/pretrained-models`, and
    `pretrained_model_name_or_path` is set to `bert-base-uncased`, we expect to find the following files:
        s3://my-bucket/bert-base-uncased/
            - pytorch_model.bin
            - config.json

    If the `LUDWIG_PRETRAINED_MODELS_DIR` environment variable is not set, we download the model from the HF hub.
    """
    pretrained_models_dir = os.environ.get("LUDWIG_PRETRAINED_MODELS_DIR")
    if pretrained_models_dir:
        pretrained_model_path = os.path.join(pretrained_models_dir, pretrained_model_name_or_path)
        if path_exists(pretrained_model_path):
            try:
                logger.info(
                    f"Found existing pretrained model artifact {pretrained_model_name_or_path} in directory "
                    f"{pretrained_models_dir}. Downloading."
                )
                return (
                    _load_pretrained_hf_model_from_dir(model_class, pretrained_model_path, **pretrained_kwargs),
                    False,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download pretrained model from {pretrained_models_dir} with error {e}. "
                    "Falling back to HuggingFace model hub."
                )

    # Fallback to HF hub.
    return load_pretrained_hf_model_from_hub(model_class, pretrained_model_name_or_path, **pretrained_kwargs), True


def upload_folder_to_hfhub(
    repo_id: str,
    folder_path: str,
    repo_type: str | None = "model",
    private: bool | None = False,
    path_in_repo: str | None = None,  # defaults to root of repo
    commit_message: str | None = None,
    commit_description: str | None = None,
) -> None:
    """Uploads a local folder to the Hugging Face Model Hub.

    Args:
        repo_id (str): The ID of the target repository on the Hugging Face Model Hub.
        folder_path (str): The local path to the folder to be uploaded.
        repo_type (str, optional): The type of the repository ('model', 'dataset', or 'space').
            Defaults to 'model'.
        private (bool, optional): If True, the repository will be private; otherwise, it will be public.
            Defaults to False.
        path_in_repo (str, optional): The relative path within the repository where the folder should be uploaded.
            Defaults to None, which means the root of the repository.
        commit_message (str, optional): A message for the commit associated with the upload.
        commit_description (str, optional): A description for the commit associated with the upload.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        ValueError: If the specified folder is empty, a file, or if an invalid 'repo_type' is provided.
        ValueError: If the upload process fails for any reason.

    Returns:
        None
    """
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")

    # Make sure the folder is not a file
    if os.path.isfile(folder_path):
        raise ValueError(f"Folder {folder_path} is a file. Please provide a folder.")

    # Make sure the folder is not empty
    if not os.listdir(folder_path):
        raise ValueError(f"Folder {folder_path} is empty.")

    if repo_type not in {"model", "dataset", "space"}:
        raise ValueError(f"Invalid repo_type {repo_type}. Valid values are 'model', 'dataset', and 'space'.")

    # Login to the hub
    api = hf_hub_login()

    # Create the repo if it doesn't exist. This is a no-op if the repo already exists
    # This is required because the API doesn't allow uploading to a non-existent repo
    if not api.repo_exists(repo_id, repo_type=repo_type):
        logger.info(f"{repo_id} does not exist. Creating.")
        api.create_repo(repo_id, private=private, exist_ok=True, repo_type=repo_type)

    # Upload the folder
    try:
        logger.info(f"Uploading folder {folder_path} to repo {repo_id}.")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
        )
    except Exception as e:
        raise ValueError(f"Failed to upload folder {folder_path} to repo {repo_id}") from e
