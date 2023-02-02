from os import PathLike
from typing import Optional, Type, Union

from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from ludwig.utils.error_handling_utils import default_retry


@default_retry()
def load_pretrained_hf_model(
    modelClass: Type,
    pretrained_model_name_or_path: Optional[Union[str, PathLike]],
    **pretrained_kwargs,
) -> PreTrainedModel:
    """Download a HuggingFace model.

    Downloads a model from the HuggingFace zoo with retry on failure.
    Args:
        modelClass: Class of the model to download.
        pretrained_model_name_or_path: Name of the model to download.
        pretrained_kwargs: Additional arguments to pass to the model constructor.
    Returns:
        The pretrained model object.
    """
    return modelClass.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)


@default_retry()
def load_pretrained_hf_tokenizer(
    pretrained_model_name_or_path: Optional[Union[str, PathLike]], **pretrained_kwargs
) -> PreTrainedTokenizer:
    """Download a HuggingFace tokenizer.

    Args:
        pretrained_model_name_or_path: Name of the tokenizer to download.
        pretrained_kwargs: Additional arguments to pass to the tokenizer constructor.
    Returns:
        The pretrained tokenizer object.
    """
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
