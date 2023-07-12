from typing import List, Tuple

import numpy as np
import pytest
import torch
import torchtext

from ludwig.constants import ENCODER_OUTPUT, LAST_HIDDEN, LOGITS, SEQUENCE, TEXT, TYPE
from ludwig.features.sequence_feature import _SequencePreprocessing, SequenceInputFeature, SequenceOutputFeature
from ludwig.features.text_feature import TextInputFeature, TextOutputFeature
from ludwig.schema.features.sequence_feature import SequenceInputFeatureConfig, SequenceOutputFeatureConfig
from ludwig.schema.features.text_feature import ECDTextInputFeatureConfig, ECDTextOutputFeatureConfig
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.utils import ENCODERS, sequence_feature

DEVICE = get_torch_device()
BATCH_SIZE = 8
SEQ_SIZE = 6
VOCAB_SIZE = 64


@pytest.fixture(scope="module")
def input_sequence() -> Tuple[torch.Tensor, List]:
    # generates a realistic looking synthetic sequence tensor, i.e.
    # each sequence will have non-zero tokens at the beginning with
    # trailing zero tokens, including a max length token with a single
    # zero token at the end.  Example:
    # [
    #   [3, 5, 6, 0, 0, 0],
    #   [10, 11, 12, 13, 14, 0],   # max length sequence
    #   [32, 0, 0, 0, 0, 0]        # minimum length sequence
    # ]
    input_tensor = torch.zeros([BATCH_SIZE, SEQ_SIZE], dtype=torch.int32).to(DEVICE)
    sequence_lengths = np.random.randint(1, SEQ_SIZE, size=BATCH_SIZE)
    for i in range(input_tensor.shape[0]):
        input_tensor[i, : sequence_lengths[i]] = torch.tensor(
            np.random.randint(2, VOCAB_SIZE, size=sequence_lengths[i])
        )

    # emulate idx2str structure
    idx2str = ["<PAD>", "<UNK>"] + [str(i) for i in range(2, VOCAB_SIZE)]

    return input_tensor, idx2str


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("sequence_type", [SEQUENCE, TEXT])
def test_sequence_input_feature(input_sequence: tuple, encoder: str, sequence_type: str):
    # test assumes "sequence data" has been tokenized and converted to
    # numeric representation.  Focus of this test is primarily on
    # integration with encoder with correctly sized encoder tensor and
    # required properties are present

    input_sequence, idx2str = input_sequence

    # setup input sequence feature definition
    # use sequence_feature() to generate baseline
    # sequence definition and then augment with
    # pre-processing metadata parameters
    input_feature_def = sequence_feature(
        encoder={
            "type": encoder,
            "max_len": SEQ_SIZE,
            # augment with emulated pre-processing metadata
            "max_sequence_length": SEQ_SIZE,
            "vocab": idx2str,
        }
    )
    input_feature_def[TYPE] = sequence_type

    # create sequence input feature object
    feature_cls = SequenceInputFeature if sequence_type == SEQUENCE else TextInputFeature
    schema_cls = SequenceInputFeatureConfig if sequence_type == SEQUENCE else ECDTextInputFeatureConfig
    sequence_config = schema_cls.from_dict(input_feature_def)
    input_feature_obj = feature_cls(sequence_config).to(DEVICE)

    # confirm dtype property
    assert input_feature_obj.input_dtype == torch.int32

    # confirm input_shape property
    assert input_feature_obj.input_shape == (SEQ_SIZE,)

    # confirm output_shape property default output shape
    # from sequence_feature() function
    encoder_output = input_feature_obj(input_sequence)
    assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, *input_feature_obj.output_shape)


@pytest.mark.parametrize("sequence_type", [SEQUENCE, TEXT])
def test_sequence_output_feature(sequence_type: str):
    output_feature_def = sequence_feature(
        decoder={
            "type": "generator",
            "max_len": SEQ_SIZE,
            "max_sequence_length": SEQ_SIZE,
            "vocab_size": VOCAB_SIZE,
        },
        input_size=VOCAB_SIZE,
    )
    output_feature_def[TYPE] = sequence_type

    feature_cls = SequenceOutputFeature if sequence_type == SEQUENCE else TextOutputFeature
    schema_cls = SequenceOutputFeatureConfig if sequence_type == SEQUENCE else ECDTextOutputFeatureConfig
    sequence_config = schema_cls.from_dict(output_feature_def)
    output_feature_obj = feature_cls(sequence_config, {}).to(DEVICE)
    combiner_outputs = {
        "combiner_output": torch.randn([BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE], dtype=torch.float32).to(DEVICE)
    }

    text_output = output_feature_obj(combiner_outputs, {})

    assert LAST_HIDDEN in text_output
    assert LOGITS in text_output
    assert text_output[LOGITS].size() == torch.Size([BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE])


def test_sequence_preproc_module_bad_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "dutch_lemmatize",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {"<EOS>": 0, "<SOS>": 1, "<PAD>": 2, "<UNK>": 3, "▁hell": 4, "o": 5, "▁world": 6},
    }

    with pytest.raises(ValueError):
        _SequencePreprocessing(metadata)


def test_sequence_preproc_module_space_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "space",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "hello": 4,
            "world": 5,
            "paleontology": 6,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["    paleontology", "unknown", "hello    world hello", "hello world hello     world    "])

    assert torch.allclose(
        res, torch.tensor([[1, 6, 0, 2, 2, 2], [1, 3, 0, 2, 2, 2], [1, 4, 5, 4, 0, 2], [1, 4, 5, 4, 5, 0]])
    )


def test_text_preproc_module_space_punct_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "space_punct",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "this": 4,
            "sentence": 5,
            "has": 6,
            "punctuation": 7,
            ",": 8,
            ".": 9,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["punctuation", ",,,,", "this... this... punctuation", "unknown"])

    assert torch.allclose(
        res, torch.tensor([[1, 7, 0, 2, 2, 2], [1, 8, 8, 8, 8, 0], [1, 4, 9, 9, 9, 4], [1, 3, 0, 2, 2, 2]])
    )


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 12, 0), reason="requires torchtext 0.12.0 or higher"
)
def test_sequence_preproc_module_sentencepiece_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "sentencepiece",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "▁hell": 4,
            "o": 5,
            "▁world": 6,
            "▁pale": 7,
            "ont": 8,
            "ology": 9,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["paleontology", "unknown", "hello world hello", "hello world hello world"])

    assert torch.allclose(
        res, torch.tensor([[1, 7, 8, 9, 0, 2], [1, 3, 3, 3, 0, 2], [1, 4, 5, 6, 4, 5], [1, 4, 5, 6, 4, 5]])
    )


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 12, 0), reason="requires torchtext 0.12.0 or higher"
)
def test_sequence_preproc_module_clip_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "clip",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "hello</w>": 4,
            "world</w>": 5,
            "pale": 7,
            "ontology</w>": 8,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["paleontology", "unknown", "hello world hello", "hello world hello world"])

    assert torch.allclose(
        res, torch.tensor([[1, 7, 8, 0, 2, 2], [1, 3, 0, 2, 2, 2], [1, 4, 5, 4, 0, 2], [1, 4, 5, 4, 5, 0]])
    )


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 12, 0), reason="requires torchtext 0.12.0 or higher"
)
def test_sequence_preproc_module_gpt2bpe_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "gpt2bpe",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "hello": 4,
            "Ġworld": 5,
            "Ġhello": 7,
            "p": 8,
            "ale": 9,
            "ont": 10,
            "ology": 11,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["paleontology", "unknown", "hello world hello", "hello world hello world"])

    assert torch.allclose(
        res, torch.tensor([[1, 8, 9, 10, 11, 0], [1, 3, 0, 2, 2, 2], [1, 4, 5, 7, 0, 2], [1, 4, 5, 7, 5, 0]])
    )


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 13, 0), reason="requires torchtext 0.13.0 or higher"
)
def test_sequence_preproc_module_bert_tokenizer():
    metadata = {
        "preprocessing": {
            "lowercase": True,
            "tokenizer": "bert",
            "unknown_symbol": "<UNK>",
            "padding_symbol": "<PAD>",
            "computed_fill_value": "<UNK>",
        },
        "max_sequence_length": SEQ_SIZE,
        "str2idx": {
            "<EOS>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
            "hello": 4,
            "world": 5,
            "pale": 7,
            "##ont": 8,
            "##ology": 9,
        },
    }
    module = _SequencePreprocessing(metadata)

    res = module(["paleontology", "unknown", "hello world hello", "hello world hello world"])

    assert torch.allclose(
        res, torch.tensor([[1, 7, 8, 9, 0, 2], [1, 3, 0, 2, 2, 2], [1, 4, 5, 4, 0, 2], [1, 4, 5, 4, 5, 0]])
    )
