import pandas as pd
import torch
from transformers import AutoTokenizer

from ludwig.backend import LocalBackend
from ludwig.constants import LOGITS, PREDICTIONS, PROBABILITIES
from ludwig.features import text_feature

TEST_MODEL_NAME = "hf-internal-testing/tiny-random-OPTForCausalLM"


def test_backwards_compatibility():
    # Tests that legacy level-based metadata keys are supported.
    metadata = {
        "SibSp": {
            "char_idx2str": ["<EOS>", "<SOS>", "<PAD>", "<UNK>", "0", "1", "2", "4", "3", "8", "5"],
            "char_max_sequence_length": 3,
            "char_pad_idx": 2,
            "char_pad_symbol": "<PAD>",
            "char_str2freq": {
                "0": 608,
                "1": 209,
                "2": 28,
                "3": 16,
                "4": 18,
                "5": 5,
                "8": 7,
                "<EOS>": 0,
                "<PAD>": 0,
                "<SOS>": 0,
                "<UNK>": 0,
            },
            "char_str2idx": {
                "0": 4,
                "1": 5,
                "2": 6,
                "3": 8,
                "4": 7,
                "5": 10,
                "8": 9,
                "<EOS>": 0,
                "<PAD>": 2,
                "<SOS>": 1,
                "<UNK>": 3,
            },
            "char_unk_symbol": "<UNK>",
            "char_vocab_size": 11,
            "preprocessing": {
                "char_most_common": 70,
                "char_sequence_length_limit": 1024,
                "char_tokenizer": "characters",
                "char_vocab_file": None,
                "computed_fill_value": "<UNK>",
                "fill_value": "<UNK>",
                "lowercase": True,
                "missing_value_strategy": "fill_with_const",
                "padding": "right",
                "padding_symbol": "<PAD>",
                "pretrained_model_name_or_path": None,
                "unknown_symbol": "<UNK>",
                "word_most_common": 20000,
                "word_sequence_length_limit": 256,
                "word_tokenizer": "space_punct",
                "word_vocab_file": None,
            },
            "word_idx2str": ["<EOS>", "<SOS>", "<PAD>", "<UNK>", "0", "1", "2", "4", "3", "8", "5"],
            "word_max_sequence_length": 3,
            "word_pad_idx": 2,
            "word_pad_symbol": "<PAD>",
            "word_str2freq": {
                "0": 608,
                "1": 209,
                "2": 28,
                "3": 16,
                "4": 18,
                "5": 5,
                "8": 7,
                "<EOS>": 0,
                "<PAD>": 0,
                "<SOS>": 0,
                "<UNK>": 0,
            },
            "word_str2idx": {
                "0": 4,
                "1": 5,
                "2": 6,
                "3": 8,
                "4": 7,
                "5": 10,
                "8": 9,
                "<EOS>": 0,
                "<PAD>": 2,
                "<SOS>": 1,
                "<UNK>": 3,
            },
            "word_unk_symbol": "<UNK>",
            "word_vocab_size": 11,
        }
    }

    column = pd.core.series.Series(["hello world", "hello world"])

    feature_data = text_feature.TextInputFeature.feature_data(
        column, metadata["SibSp"], metadata["SibSp"]["preprocessing"], LocalBackend()
    )

    assert list(feature_data[0]) == [1, 3, 3]
    assert list(feature_data[1]) == [1, 3, 3]


def test_get_decoded_targets_and_predictions():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    vocab_size = 8

    # Scenario 1: Prediction and target tensors have the same length, so nothing should change
    targets = torch.tensor([[78, 79, 504, 76, 397, 84, 0]])
    predictions = {
        PREDICTIONS: torch.tensor([[78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64),
        PROBABILITIES: torch.randn(1, 7, vocab_size).to(torch.float32),
        LOGITS: torch.randn(1, 7, vocab_size).to(torch.float32),
    }
    (
        decoded_targets,
        decoded_predictions,
    ) = text_feature.get_decoded_targets_and_predictions(targets, predictions, tokenizer)

    assert decoded_targets == [" first she 18 yearman our<s>"]
    assert decoded_predictions == [" first she 18 yearman our<s>"]
