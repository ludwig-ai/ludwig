from os import PathLike
from typing import Optional, Type, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ludwig.utils.error_handling_utils import default_retry


@default_retry()
def load_pretrained_hf_model(
    modelClass: Type, pretrained_model_name_or_path: Optional[Union[str, PathLike]], **pretrained_kwargs
) -> PreTrainedTokenizerBase:
    """Download a HuggingFace model.

    Downloads a model from the HuggingFace zoo with retry on failure.

    Args:
        model: Class of the model to download.

    Returns:
        The pretrained model object.
    """
    return modelClass.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
