import os
from typing import Type

import pytest
from transformers import (  # CamembertModel,
    AlbertModel,
    BertModel,
    BertTokenizer,
    CTRLModel,
    DistilBertModel,
    ElectraModel,
    FlaubertModel,
    GPT2Model,
    LongformerModel,
    MT5EncoderModel,
    OpenAIGPTModel,
    RobertaModel,
    T5Model,
    TransfoXLModel,
    XLMModel,
    XLMRobertaModel,
    XLNetModel,
)

from ludwig.encoders.text_encoders import (  # CamemBERTEncoder,
    ALBERTEncoder,
    BERTEncoder,
    CTRLEncoder,
    DistilBERTEncoder,
    ELECTRAEncoder,
    FlauBERTEncoder,
    GPT2Encoder,
    GPTEncoder,
    LongformerEncoder,
    MT5Encoder,
    RoBERTaEncoder,
    T5Encoder,
    TransformerXLEncoder,
    XLMEncoder,
    XLMRoBERTaEncoder,
    XLNetEncoder,
)
from ludwig.utils.hf_utils import load


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME),
        # (AutoModel, AutoTransformerEncoder.DEFAULT_MODEL_NAME),
        # (AutoTokenizer, None),
        (BertModel, BERTEncoder.DEFAULT_MODEL_NAME),
        (BertTokenizer, "bert-base-uncased"),
        # (CamembertModel, CamemBERTEncoder.DEFAULT_MODEL_NAME),
        (CTRLModel, CTRLEncoder.DEFAULT_MODEL_NAME),
        (DistilBertModel, DistilBERTEncoder.DEFAULT_MODEL_NAME),
        (ElectraModel, ELECTRAEncoder.DEFAULT_MODEL_NAME),
        (FlaubertModel, FlauBERTEncoder.DEFAULT_MODEL_NAME),
        (GPT2Model, GPT2Encoder.DEFAULT_MODEL_NAME),
        (LongformerModel, LongformerEncoder.DEFAULT_MODEL_NAME),
        (MT5EncoderModel, MT5Encoder.DEFAULT_MODEL_NAME),
        (OpenAIGPTModel, GPTEncoder.DEFAULT_MODEL_NAME),
        (RobertaModel, RoBERTaEncoder.DEFAULT_MODEL_NAME),
        (T5Model, T5Encoder.DEFAULT_MODEL_NAME),
        (TransfoXLModel, TransformerXLEncoder.DEFAULT_MODEL_NAME),
        (XLMModel, XLMEncoder.DEFAULT_MODEL_NAME),
        (XLMRobertaModel, XLMRoBERTaEncoder.DEFAULT_MODEL_NAME),
        (XLNetModel, XLNetEncoder.DEFAULT_MODEL_NAME),
    ],
)
def test_load_hf_model(model: Type, name: str, tmpdir: os.PathLike):
    """Ensure that the HF models used in ludwig download correctly."""
    cache_dir = os.path.join(tmpdir, name.replace(os.path.sep, "_") if name else str(model.__name__))
    os.makedirs(cache_dir, exist_ok=True)
    loaded_model = load(model, name, cache_dir=cache_dir, force_download=True)
    assert isinstance(loaded_model, model)
    assert os.listdir(cache_dir)
