import pytest
import torch
import torchtext
from transformers.models.bert.tokenization_bert import PRETRAINED_INIT_CONFIGURATION, PRETRAINED_VOCAB_FILES_MAP

from ludwig.utils.tokenizers import SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        pytest.param(
            model_name,
            marks=[
                pytest.mark.skipif(
                    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 13, 0),
                    reason="requires torchtext 0.13.0 or higher",
                ),
                pytest.mark.skipif(model_name in SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES, reason="issue on torchtext side"),
            ],
        )
        for model_name in PRETRAINED_VOCAB_FILES_MAP["vocab_file"].keys()
    ],
)
def test_bert_hf_tokenizer_parity(pretrained_model_name_or_path):
    from ludwig.utils.tokenizers import BERTTokenizer, get_hf_tokenizer, HFTokenizer

    inputs = "Hello, I'm a single sentence!"
    hf_tokenizer = HFTokenizer(pretrained_model_name_or_path)
    tokens_expected = hf_tokenizer.tokenizer.tokenize(inputs)
    token_ids_expected = hf_tokenizer(inputs)

    vocab_file = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][pretrained_model_name_or_path]
    init_kwargs = PRETRAINED_INIT_CONFIGURATION[pretrained_model_name_or_path]
    tokenizer = BERTTokenizer(vocab_file, **init_kwargs)
    tokens = tokenizer(inputs)

    tokenizer_ids_only = get_hf_tokenizer(pretrained_model_name_or_path)
    token_ids = tokenizer_ids_only(inputs)

    assert not isinstance(tokenizer_ids_only, HFTokenizer)
    assert tokens == tokens_expected
    assert token_ids == token_ids_expected
