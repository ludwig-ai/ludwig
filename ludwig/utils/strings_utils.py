#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import re
import unicodedata
from collections import Counter
from enum import Enum
from typing import List, Optional, Set, Union

import numpy as np

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.data.dataframe.pandas import PANDAS
from ludwig.utils.fs_utils import open_file
from ludwig.utils.math_utils import int_type
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.tokenizers import tokenizer_registry
from ludwig.utils.types import Series

BOOL_TRUE_STRS = {"yes", "y", "true", "t", "1", "1.0"}
BOOL_FALSE_STRS = {"no", "n", "false", "f", "0", "0.0"}

# Special symbols.
STOP_SYMBOL = "<EOS>"
START_SYMBOL = "<SOS>"
PADDING_SYMBOL = "<PAD>"
UNKNOWN_SYMBOL = "<UNK>"


class SpecialSymbol(Enum):
    """Special symbols used for text features."""

    STOP = 0
    START = 1
    PADDING = 2
    UNKNOWN = 3


def all_bool_strs():
    """Returns all valid boolean strings, with varied capitalization."""
    fns = [lambda x: x, lambda x: x.upper(), lambda x: x.capitalize()]
    return sorted({fn(x) for fn in fns for x in BOOL_TRUE_STRS | BOOL_FALSE_STRS})


def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def str2bool(v: str, fallback_true_label=None) -> bool:
    """Returns bool representation of the given value v.

    Check the value against global bool string lists.
    Fallback to using fallback_true_label as True if the value isn't in the global bool lists.

    args:
        v: Value to get the bool representation for.
        fallback_true_label: (str) label to use as 'True'.
    """
    v_str = str(v).lower()
    if v_str in BOOL_TRUE_STRS:
        return True
    if v_str in BOOL_FALSE_STRS:
        return False
    if fallback_true_label is None:
        raise ValueError(f"Cannot automatically map value {v} to a boolean and no `fallback_true_label` specified")
    return v == fallback_true_label


def are_conventional_bools(values: List[Union[str, bool]]) -> bool:
    """Returns whether all values are conventional booleans."""
    for value in values:
        lower_value = str(value).lower()
        if lower_value not in BOOL_TRUE_STRS and lower_value not in BOOL_FALSE_STRS:
            return False
    return True


def is_number(s: Union[str, int, float]):
    """Returns whether specified value is number."""
    if isinstance(s, str) and s.lower() == "nan":
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def are_all_numbers(values: List[Union[str, int, float]]):
    """Returns whether all values are numbers."""
    for value in values:
        if not is_number(value):
            return False
    return True


def is_integer(s: Union[str, int, float]):
    """Returns whether specified value is an integer."""
    try:
        float(s)
    except ValueError:
        return False
    else:
        return float(s).is_integer() and not np.isnan(float(s))


def are_sequential_integers(values: List[Union[str, int, float]]):
    """Returns whether distinct values form sequential integer list."""
    int_list = []
    for value in values:
        if not is_integer(value):
            return False
        int_list.append(int(float(value)))
    return (max(int_list) - min(int_list) + 1) == len(int_list)


def match_replace(string_to_match, list_regex):
    """Matches strings against regular expressions.

    arguments:
    string_to_match -- the string to match

    returns:
    string_to_match -- the cleaned string
    matched -- the list of regular expressions that matched
    """
    matched = []
    for regex in list_regex:
        match = re.search(regex[0], string_to_match)
        if match:
            string_to_match = re.sub(regex[0], regex[1], string_to_match)
            matched.append(regex[0].pattern)
    return string_to_match, matched


def load_vocabulary(vocab_file):
    with open_file(vocab_file, "r", encoding="utf-8") as f:
        vocabulary = []
        for line in f:
            line = line.strip()
            if " " in line:
                line = line.split(" ")[0]
            vocabulary.append(line)
        return vocabulary


def add_or_move_symbol(vocab_list: List[str], vocab_set: Set[str], symbol: str, index: int):
    """Inserts or moves the symbol to the specified index."""
    if symbol in vocab_set:
        vocab_list.remove(symbol)
    vocab_list.insert(index, symbol)


def create_vocabulary(
    data: Series,
    tokenizer_type: str = "space",
    lowercase: bool = True,
    num_most_frequent: int = None,
    vocab_file: str = None,
    add_special_symbols: bool = True,
    unknown_symbol: str = UNKNOWN_SYMBOL,
    padding_symbol: str = PADDING_SYMBOL,
    start_symbol: str = START_SYMBOL,
    stop_symbol: str = STOP_SYMBOL,
    pretrained_model_name_or_path: str = None,
    processor: DataFrameEngine = PANDAS,
):
    """Computes a vocabulary over the provided data frame.

    This function is used when the data consists of multiple tokens within one example. E.g., words in a text feature,
    items in a set feature, etc. If the feature only contains a single token, use `create_vocabulary_single_token`
    instead.

    A tokenizer is specified using the `tokenizer_type`. The tokenizer will be used to process all of the data
    provided, producing an indexed vocabulary with frequency counts. If the `tokenizer_type` is 'hf_tokenizer',
    then a pre-trained huggingface tokenizer is loaded from `pretrained_model_name_or_path` and that vocabulary is
    used directly.

    The UNKNOWN special symbol is always included in the final vocabulary. Additional special symbols (PADDING, START,
    STOP) are added if add_special_symbols=True.

    Args:
        data: Series of string data.
        tokenizer_type: Tokenizer type. Can be a tokenizer registry value or 'hf_tokenizer' for huggingface.
        lowercase: Whether to lowercase all strings.
        num_most_frequent: Upper limit on vocabulary size.,
        add_special_symbols: If True, START, STOP, PADDING special symbols are added to the vocabulary. UNKNOWN is
            always added.
        unknown_symbol: String representation for the UNKNOWN symbol.
        padding_symbol: String representation for the PADDING symbol.
        start_symbol: String representation for the START symbol.
        stop_symbol: String representation for the STOP symbol.
        pretrained_model_name_or_path: Name/path to huggingface model.
        processor: Which processor to use to process data.

    Returns:
        Tuple of:
            vocab: List of strings representing the computed vocabulary.
            str2idx: Map of symbol to index.
            str2freq: Map of symbol to frequency.
            line_length_max: (int) maximum sequence length.
            line_length_99ptile: (float) 99th percentile of maximum sequence length.
            pad_idx: Index to padding symbol.
            padding_symbol: Actual padding symbol.
            unknown_symbol: Actual unknown symbol.

    TODO(Justin): Clean up pad_idx, padding_symbol, unknown_symbol return, as no one seems to be using it.
    """
    vocab = None

    tokenizer = get_from_registry(tokenizer_type, tokenizer_registry)(
        vocab_file=vocab_file,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    # Pre-trained huggingface tokenizer. Use the pre-existing vocabulary and special symbols.
    if tokenizer_type == "hf_tokenizer":
        try:
            vocab = tokenizer.tokenizer.get_vocab()
            vocab = list(vocab.keys())
        except NotImplementedError:
            vocab = []
            for idx in range(tokenizer.tokenizer.vocab_size):
                vocab.append(tokenizer.tokenizer._convert_id_to_token(idx))
            vocab += tokenizer.tokenizer.added_tokens_encoder.keys()

        pad_token = tokenizer.tokenizer.pad_token
        unk_token = tokenizer.tokenizer.unk_token

        if unk_token is None:
            vocab = [unknown_symbol] + vocab
        else:
            unknown_symbol = unk_token

        if pad_token is None and add_special_symbols:
            vocab = [padding_symbol] + vocab
        else:
            padding_symbol = pad_token
    elif hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif vocab_file is not None:
        vocab = load_vocabulary(vocab_file)

    processed_lines = data.map(lambda line: tokenizer(line.lower() if lowercase else line))
    processed_counts = processed_lines.explode().value_counts(sort=False)
    processed_counts = processor.compute(processed_counts)
    unit_counts = Counter(dict(processed_counts))
    line_length_max = processor.compute(processed_lines.map(len).max())
    line_length_99ptile = processor.compute(processed_lines.map(len).quantile(0.99))

    if vocab is None:
        vocab = [unit for unit, count in unit_counts.most_common(num_most_frequent)]

    vocab_set = set(vocab)

    if tokenizer_type != "hf_tokenizer":
        if add_special_symbols:
            add_or_move_symbol(vocab, vocab_set, stop_symbol, SpecialSymbol.STOP.value)
            add_or_move_symbol(vocab, vocab_set, start_symbol, SpecialSymbol.START.value)
            add_or_move_symbol(vocab, vocab_set, padding_symbol, SpecialSymbol.PADDING.value)
        # Always add the UNKNOWN symbol if we're using our own tokenizer.
        add_or_move_symbol(vocab, vocab_set, unknown_symbol, SpecialSymbol.UNKNOWN.value)

    str2idx = {unit: i for i, unit in enumerate(vocab)}
    str2freq = {unit: unit_counts.get(unit) if unit in unit_counts else 0 for unit in vocab}

    pad_idx = None
    if padding_symbol in str2idx.keys():
        pad_idx = str2idx[padding_symbol]

    return vocab, str2idx, str2freq, line_length_max, line_length_99ptile, pad_idx, padding_symbol, unknown_symbol


def create_vocabulary_single_token(
    data: Series,
    num_most_frequent: Optional[int] = None,
    processor: DataFrameEngine = PANDAS,
    unknown_symbol: str = UNKNOWN_SYMBOL,
):
    """Computes a vocabulary over the provided data frame.

    This function should be used iff the values in each row of data should be considered as a single token, e.g.,
    category features ("interested", "not interested", "somewhat interested").

    This assumption allows us to be more efficient than `create_vocabulary()` as we can skip tokenization and
    computing the maximum sequence length, which are unnecessary for category features.

    The `UNKNOWN` special symbol is always included in the final vocabulary. Additional special symbols (PADDING, START,
    STOP) are added if add_special_symbols=True.

    Args:
        data: Series of string data.
        num_most_frequent: Upper limit on vocabulary size.
        unknown_symbol: String representation for the UNKNOWN symbol.
        processor: Which processor to use to process data.

    Returns:
        Tuple of:
            vocab: List of strings representing the computed vocabulary.
            str2idx: Map of symbol to index.
            str2freq: Map of symbol to frequency.
    """
    processed_counts = data.str.strip().value_counts(sort=True)
    processed_counts = processor.compute(processed_counts)
    vocab = [unknown_symbol] + processed_counts.index.tolist()[:num_most_frequent]
    str2idx = {unit: i for i, unit in enumerate(vocab)}
    str2freq = processed_counts.to_dict()
    str2freq = {k: str2freq.get(k, 0) for k in vocab}
    return vocab, str2idx, str2freq


def _get_sequence_vector(
    sequence, tokenizer, tokenizer_type, format_dtype, unit_to_id, lowercase=True, unknown_symbol=UNKNOWN_SYMBOL
) -> np.ndarray:
    unit_sequence = tokenizer(sequence.lower() if lowercase else sequence)

    unit_indices_vector = np.empty(len(unit_sequence), dtype=format_dtype)
    for i in range(len(unit_sequence)):
        curr_unit = unit_sequence[i]
        if tokenizer_type == "hf_tokenizer":
            unit_indices_vector[i] = curr_unit
        else:
            if curr_unit in unit_to_id:
                unit_indices_vector[i] = unit_to_id[curr_unit]
            else:
                unit_indices_vector[i] = unit_to_id[unknown_symbol]

    # Add start and stop symbols.
    # Huggingface's pretrained tokenizers take care of this implicitly:
    # https://huggingface.co/docs/transformers/preprocessing
    if tokenizer_type != "hf_tokenizer":
        unit_indices_vector = np.append(unit_indices_vector, unit_to_id[STOP_SYMBOL])
        unit_indices_vector = np.insert(unit_indices_vector, 0, unit_to_id[START_SYMBOL])
    return unit_indices_vector


def build_sequence_matrix(
    sequences,  # pd.core.series.Series
    inverse_vocabulary,
    tokenizer_type,
    length_limit,
    padding_symbol=PADDING_SYMBOL,
    padding="right",
    unknown_symbol=UNKNOWN_SYMBOL,
    lowercase=True,
    tokenizer_vocab_file=None,
    pretrained_model_name_or_path=None,
    processor=PANDAS,
) -> np.ndarray:
    tokenizer = get_from_registry(tokenizer_type, tokenizer_registry)(
        vocab_file=tokenizer_vocab_file,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    format_dtype = int_type(len(inverse_vocabulary) - 1)

    unit_vectors = sequences.map(
        lambda sequence: _get_sequence_vector(
            sequence,
            tokenizer,
            tokenizer_type,
            format_dtype,
            inverse_vocabulary,
            lowercase=lowercase,
            unknown_symbol=unknown_symbol,
        )
    )

    max_length = processor.compute(unit_vectors.map(len).max())
    if max_length < length_limit:
        logging.debug(f"max length of {format}: {max_length} < limit: {length_limit}")
    max_length = length_limit

    def pad(vector):
        sequence = np.full((int(max_length),), inverse_vocabulary[padding_symbol], dtype=format_dtype)
        limit = min(vector.shape[0], max_length)
        if padding == "right":
            sequence[:limit] = vector[:limit]
        else:  # if padding == 'left
            sequence[max_length - limit :] = vector[:limit]
        return sequence

    padded = processor.map_objects(unit_vectors, pad)
    return padded
