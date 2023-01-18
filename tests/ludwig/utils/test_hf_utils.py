import os
from typing import Type

import pytest
from transformers import BertModel, BertTokenizer

from ludwig.encoders.text_encoders import BERTEncoder
from ludwig.utils.hf_utils import load_pretrained_hf_model


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (BertModel, BERTEncoder.DEFAULT_MODEL_NAME),
        (BertTokenizer, "bert-base-uncased"),
    ],
)
def test_load_hf_model(model: Type, name: str, tmpdir: os.PathLike):
    """Ensure that the HF models used in ludwig download correctly."""
    cache_dir = os.path.join(tmpdir, name.replace(os.path.sep, "_") if name else str(model.__name__))
    os.makedirs(cache_dir, exist_ok=True)
    loaded_model = load_pretrained_hf_model(model, name, cache_dir=cache_dir, force_download=True)
    assert isinstance(loaded_model, model)
    assert os.listdir(cache_dir)
