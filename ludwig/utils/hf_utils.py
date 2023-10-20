import logging
import os
import tempfile
from os import PathLike
from typing import Optional, Tuple, Type, Union

from transformers import PreTrainedModel

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.error_handling_utils import default_retry
from ludwig.utils.fs_utils import download, path_exists

logger = logging.getLogger(__name__)


@default_retry()
def _load_pretrained_hf_class_from_hub(
    hf_class: Type,
    pretrained_model_name_or_path: Optional[Union[str, PathLike]],
    **pretrained_kwargs,
) -> PreTrainedModel:
    """Download a HuggingFace artifact (model, tokenizer, config).

    Downloads a model from the HuggingFace zoo with retry on failure.
    Args:
        hf_class: Class of the model to download.
        pretrained_model_name_or_path: Name of the model to download.
        pretrained_kwargs: Additional arguments to pass to the model constructor.
    Returns:
        The pretrained model object.
    """
    return hf_class.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)


def _load_pretrained_hf_class_from_dir(
    hf_class: Type,
    pretrained_model_name_or_path: Optional[Union[str, PathLike]],
    **pretrained_kwargs,
) -> PreTrainedModel:
    """Downloads a model to a local temporary directory, and Loads a pretrained HF model from a local directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        download(pretrained_model_name_or_path, tmpdir)
        return hf_class.from_pretrained(tmpdir, **pretrained_kwargs)


@DeveloperAPI
def load_pretrained_hf_class_with_hub_fallback(
    hf_class: Type,
    pretrained_model_name_or_path: Optional[Union[str, PathLike]],
    **pretrained_kwargs,
) -> Tuple[PreTrainedModel, bool]:
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
        logger.info("LUDWIG_PRETRAINED_MODELS_DIR was set. Attempting to load HF artifact from S3.")
        pretrained_model_path = os.path.join(pretrained_models_dir, pretrained_model_name_or_path)
        if path_exists(pretrained_model_path):
            try:
                return (
                    _load_pretrained_hf_class_from_dir(hf_class, pretrained_model_path, **pretrained_kwargs),
                    False,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download pretrained artifact for hf class {hf_class} from {pretrained_models_dir} with "
                    f"error {e}. Falling back to HuggingFace model hub."
                )

    # Fallback to HF hub.
    return _load_pretrained_hf_class_from_hub(hf_class, pretrained_model_name_or_path, **pretrained_kwargs), True
