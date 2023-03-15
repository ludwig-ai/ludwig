import os
from typing import Type

import pytest
from transformers import AlbertModel, BertModel, BertTokenizer

from ludwig.encoders.text_encoders import ALBERTEncoder, BERTEncoder
from ludwig.utils.hf_utils import load_pretrained_hf_model_from_hub, load_pretrained_hf_model_with_hub_fallback


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME),
        (BertTokenizer, "bert-base-uncased"),
    ],
)
def test_load_pretrained_hf_model_from_hub(model: Type, name: str, tmpdir: os.PathLike):
    """Ensure that the HF models used in ludwig download correctly."""
    cache_dir = os.path.join(tmpdir, name.replace(os.path.sep, "_") if name else str(model.__name__))
    os.makedirs(cache_dir, exist_ok=True)
    loaded_model = load_pretrained_hf_model_from_hub(model, name, cache_dir=cache_dir, force_download=True)
    assert isinstance(loaded_model, model)
    assert os.listdir(cache_dir)


def test_load_pretrained_hf_model_with_hub_fallback(tmpdir):
    """Ensure that the HF models used in ludwig download correctly with S3 or hub fallback."""
    # Don't set env var.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME)
    assert used_fallback

    # Download the model, load it from tmpdir, and set env var.
    load_pretrained_hf_model_from_hub(AlbertModel, "albert-base-v2").save_pretrained(
        os.path.join(tmpdir, "albert-base-v2")
    )
    os.environ["LUDWIG_PRETRAINED_MODELS_DIR"] = f"file://{tmpdir}"  # Needs to be an absolute path.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME)
    assert not used_fallback

    # Fallback is used for a model that doesn't exist in models directory.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(BertModel, BERTEncoder.DEFAULT_MODEL_NAME)
    assert used_fallback

    # Clean up.
    del os.environ["LUDWIG_PRETRAINED_MODELS_DIR"]
