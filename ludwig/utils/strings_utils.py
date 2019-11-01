#! /usr/bin/env python
# coding=utf-8
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
from abc import abstractmethod
from collections import Counter

import numpy as np
import unicodedata

from ludwig.utils.math_utils import int_type
from ludwig.utils.misc import get_from_registry
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

UNKNOWN_SYMBOL = '<UNK>'
PADDING_SYMBOL = '<PAD>'

SPLIT_REGEX = re.compile(r'\s+')
SPACE_PUNCTUATION_REGEX = re.compile(r'\w+|[^\w\s]')
COMMA_REGEX = re.compile(r'\s*,\s*')
UNDERSCORE_REGEX = re.compile(r'\s*_\s*')


def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return '_'

    return ''.join(safe_char(c) for c in s).rstrip('_')


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


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
    with open(vocab_file, 'r') as f:
        vocabulary = []
        for line in f:
            line = line.strip()
            if ' ' in line:
                line = line.split(' ')[0]
            vocabulary.append(line)
        return vocabulary
        #return [line.strip() for line in f]


def create_vocabulary(
        data,
        tokenizer_type='space',
        add_unknown=True,
        add_padding=True,
        lowercase=True,
        num_most_frequent=None,
        vocab_file=None,
        unknown_symbol=UNKNOWN_SYMBOL,
        padding_symbol=PADDING_SYMBOL
):
    vocab = None
    max_line_length = 0
    unit_counts = Counter()

    if tokenizer_type == 'bert':
        vocab = load_vocabulary(vocab_file)
        add_unknown = False
        add_padding = False
    elif vocab_file is not None:
        vocab = load_vocabulary(vocab_file)

    tokenizer = get_from_registry(
        tokenizer_type,
        tokenizer_registry
    )(vocab_file=vocab_file)
    for line in data:
        processed_line = tokenizer(line.lower() if lowercase else line)
        unit_counts.update(processed_line)
        max_line_length = max(max_line_length, len(processed_line))

    if vocab is None:
        vocab = [unit for unit, count in
                 unit_counts.most_common(num_most_frequent)]

    vocab_set = set(vocab)
    if add_unknown:
        if unknown_symbol not in vocab_set:
            vocab = [unknown_symbol] + vocab
    if add_padding:
        if padding_symbol not in vocab_set:
            vocab = [padding_symbol] + vocab

    str2idx = {unit: i for i, unit in enumerate(vocab)}
    str2freq = {unit: unit_counts.get(unit) if unit in unit_counts else 0 for
                unit in vocab}

    return vocab, str2idx, str2freq, max_line_length


def get_sequence_vector(sequence, tokenizer_type, unit_to_id, lowercase=True):
    tokenizer = get_from_registry(tokenizer_type, tokenizer_registry)()
    format_dtype = int_type(len(unit_to_id) - 1)
    return _get_sequence_vector(
        sequence,
        tokenizer,
        format_dtype,
        unit_to_id,
        lowercase=lowercase
    )


def _get_sequence_vector(
        sequence,
        tokenizer,
        format_dtype,
        unit_to_id,
        lowercase=True,
        unknown_symbol=UNKNOWN_SYMBOL
):
    unit_sequence = tokenizer(
        sequence.lower() if lowercase else sequence
    )
    unit_indices_vector = np.empty(len(unit_sequence), dtype=format_dtype)
    for i in range(len(unit_sequence)):
        curr_unit = unit_sequence[i]
        if curr_unit in unit_to_id:
            unit_indices_vector[i] = unit_to_id[curr_unit]
        else:
            unit_indices_vector[i] = unit_to_id[unknown_symbol]
    return unit_indices_vector


def build_sequence_matrix(
        sequences,
        inverse_vocabulary,
        tokenizer_type,
        length_limit,
        padding_symbol,
        padding='right',
        unknown_symbol=UNKNOWN_SYMBOL,
        lowercase=True,
        tokenizer_vocab_file=None,
):
    tokenizer = get_from_registry(tokenizer_type, tokenizer_registry)(
        vocab_file=tokenizer_vocab_file
    )
    format_dtype = int_type(len(inverse_vocabulary) - 1)

    max_length = 0
    unit_vectors = []
    for sequence in sequences:
        unit_indices_vector = _get_sequence_vector(
            sequence,
            tokenizer,
            format_dtype,
            inverse_vocabulary,
            lowercase=lowercase,
            unknown_symbol=unknown_symbol
        )
        unit_vectors.append(unit_indices_vector)
        if len(unit_indices_vector) > max_length:
            max_length = len(unit_indices_vector)

    if max_length < length_limit:
        logging.debug('max length of {0}: {1} < limit: {2}'.format(
            format, max_length, length_limit
        ))
    max_length = length_limit
    sequence_matrix = np.full((len(sequences), max_length),
                              inverse_vocabulary[padding_symbol],
                              dtype=format_dtype)
    for i, vector in enumerate(unit_vectors):
        limit = min(vector.shape[0], max_length)
        if padding == 'right':
            sequence_matrix[i, :limit] = vector[:limit]
        else:  # if padding == 'left
            sequence_matrix[i, max_length - limit:] = vector[:limit]
    return sequence_matrix


class BaseTokenizer:
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, text):
        pass


class CharactersToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return text


class SpaceStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return SPLIT_REGEX.split(text.strip())


class SpacePunctuationStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return SPACE_PUNCTUATION_REGEX.findall(text.strip())


class UnderscoreStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return UNDERSCORE_REGEX.split(text.strip())


class CommaStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return COMMA_REGEX.split(text.strip())


class UntokenizedStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return [text]


class StrippedStringToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return [text.strip()]


class EnglishTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('en'))


class EnglishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class EnglishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            filter_stopwords=True
        )


class EnglishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        process_text(text, load_nlp_pipeline('en'), return_lemma=True)


class EnglishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class EnglishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            return_lemma=True,
            filter_stopwords=True
        )


class ItalianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('it'))


class ItalianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class ItalianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            filter_stopwords=True
        )


class ItalianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True
        )


class ItalianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class ItalianLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True,
            filter_stopwords=True
        )


class SpanishTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('es'))


class SpanishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class SpanishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            filter_stopwords=True
        )


class SpanishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True
        )


class SpanishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class SpanishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True,
            filter_stopwords=True
        )


class GermanTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('de'))


class GermanFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GermanRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            filter_stopwords=True
        )


class GermanLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True
        )


class GermanLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GermanLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True,
            filter_stopwords=True
        )


class FrenchTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('fr'))


class FrenchFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class FrenchRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            filter_stopwords=True
        )


class FrenchLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True
        )


class FrenchLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class FrenchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True,
            filter_stopwords=True
        )


class PortugueseTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('pt'))


class PortugueseFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class PortugueseRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            filter_stopwords=True
        )


class PortugueseLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('pt'), return_lemma=True)


class PortugueseLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class PortugueseLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            return_lemma=True,
            filter_stopwords=True
        )


class DutchTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('nl'))


class DutchFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class DutchRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            filter_stopwords=True
        )


class DutchLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('nl'), return_lemma=True)


class DutchLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class DutchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            return_lemma=True,
            filter_stopwords=True
        )


class GreekTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('el'))


class GreekFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GreekRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            filter_stopwords=True
        )


class GreekLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('el'), return_lemma=True)


class GreekLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GreekLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            return_lemma=True,
            filter_stopwords=True
        )


class NorwegianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('nb'))


class NorwegianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class NorwegianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            filter_stopwords=True
        )


class NorwegianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('nb'), return_lemma=True)


class NorwegianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class NorwegianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            return_lemma=True,
            filter_stopwords=True
        )

class LithuanianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('lt'))


class LithuanianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class LithuanianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            filter_stopwords=True
        )


class LithuanianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('lt'), return_lemma=True)


class LithuanianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class LithuanianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            return_lemma=True,
            filter_stopwords=True
        )

class MultiTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('xx'))




class MultiFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class MultiRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            filter_stopwords=True
        )


class MultiLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline('xx'), return_lemma=True)


class MultiLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class MultiLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            return_lemma=True,
            filter_stopwords=True
        )


class BERTTokenizer(BaseTokenizer):
    def __init__(self, vocab_file=None, **kwargs):

        if vocab_file is None:
            raise ValueError(
                'Vocabulary file is required to initialize BERT tokenizer'
            )

        try:
            from bert.tokenization import FullTokenizer
        except ImportError:
            raise ValueError(
                "Please install bert-tensorflow: pip install bert-tensorflow"
            )

        self.tokenizer = FullTokenizer(vocab_file)

    def __call__(self, text):
        return ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']


tokenizer_registry = {
    'characters': CharactersToListTokenizer,
    'space': SpaceStringToListTokenizer,
    'space_punct': SpacePunctuationStringToListTokenizer,
    'underscore': UnderscoreStringToListTokenizer,
    'comma': CommaStringToListTokenizer,
    'untokenized': UntokenizedStringToListTokenizer,
    'stripped': StrippedStringToListTokenizer,
    'english_tokenize': EnglishTokenizer,
    'english_tokenize_filter': EnglishFilterTokenizer,
    'english_tokenize_remove_stopwords': EnglishRemoveStopwordsTokenizer,
    'english_lemmatize': EnglishLemmatizeTokenizer,
    'english_lemmatize_filter': EnglishLemmatizeFilterTokenizer,
    'english_lemmatize_remove_stopwords': EnglishLemmatizeRemoveStopwordsTokenizer,
    'italian_tokenize': ItalianTokenizer,
    'italian_tokenize_filter': ItalianFilterTokenizer,
    'italian_tokenize_remove_stopwords': ItalianRemoveStopwordsTokenizer,
    'italian_lemmatize': ItalianLemmatizeTokenizer,
    'italian_lemmatize_filter': ItalianLemmatizeFilterTokenizer,
    'italian_lemmatize_remove_stopwords': ItalianLemmatizeRemoveStopwordsTokenizer,
    'spanish_tokenize': SpanishTokenizer,
    'spanish_tokenize_filter': SpanishFilterTokenizer,
    'spanish_tokenize_remove_stopwords': SpanishRemoveStopwordsTokenizer,
    'spanish_lemmatize': SpanishLemmatizeTokenizer,
    'spanish_lemmatize_filter': SpanishLemmatizeFilterTokenizer,
    'spanish_lemmatize_remove_stopwords': SpanishLemmatizeRemoveStopwordsTokenizer,
    'german_tokenize': GermanTokenizer,
    'german_tokenize_filter': GermanFilterTokenizer,
    'german_tokenize_remove_stopwords': GermanRemoveStopwordsTokenizer,
    'german_lemmatize': GermanLemmatizeTokenizer,
    'german_lemmatize_filter': GermanLemmatizeFilterTokenizer,
    'german_lemmatize_remove_stopwords': GermanLemmatizeRemoveStopwordsTokenizer,
    'french_tokenize': FrenchTokenizer,
    'french_tokenize_filter': FrenchFilterTokenizer,
    'french_tokenize_remove_stopwords': FrenchRemoveStopwordsTokenizer,
    'french_lemmatize': FrenchLemmatizeTokenizer,
    'french_lemmatize_filter': FrenchLemmatizeFilterTokenizer,
    'french_lemmatize_remove_stopwords': FrenchLemmatizeRemoveStopwordsTokenizer,
    'portuguese_tokenize': PortugueseTokenizer,
    'portuguese_tokenize_filter': PortugueseFilterTokenizer,
    'portuguese_tokenize_remove_stopwords': PortugueseRemoveStopwordsTokenizer,
    'portuguese_lemmatize': PortugueseLemmatizeTokenizer,
    'portuguese_lemmatize_filter': PortugueseLemmatizeFilterTokenizer,
    'portuguese_lemmatize_remove_stopwords': PortugueseLemmatizeRemoveStopwordsTokenizer,
    'dutch_tokenize': DutchTokenizer,
    'dutch_tokenize_filter': DutchFilterTokenizer,
    'dutch_tokenize_remove_stopwords': DutchRemoveStopwordsTokenizer,
    'dutch_lemmatize': DutchLemmatizeTokenizer,
    'dutch_lemmatize_filter': DutchLemmatizeFilterTokenizer,
    'dutch_lemmatize_remove_stopwords': DutchLemmatizeRemoveStopwordsTokenizer,
    'greek_tokenize': GreekTokenizer,
    'greek_tokenize_filter': GreekFilterTokenizer,
    'greek_tokenize_remove_stopwords': GreekRemoveStopwordsTokenizer,
    'greek_lemmatize': GreekLemmatizeTokenizer,
    'greek_lemmatize_filter': GreekLemmatizeFilterTokenizer,
    'greek_lemmatize_remove_stopwords': GreekLemmatizeRemoveStopwordsFilterTokenizer,
    'norwegian_tokenize': NorwegianTokenizer,
    'norwegian_tokenize_filter': NorwegianFilterTokenizer,
    'norwegian_tokenize_remove_stopwords': NorwegianRemoveStopwordsTokenizer,
    'norwegian_lemmatize': NorwegianLemmatizeTokenizer,
    'norwegian_lemmatize_filter': NorwegianLemmatizeFilterTokenizer,
    'norwegian_lemmatize_remove_stopwords': NorwegianLemmatizeRemoveStopwordsFilterTokenizer,
    'lithuanian_tokenize': LithuanianTokenizer,
    'lithuanian_tokenize_filter': LithuanianFilterTokenizer,
    'lithuanian_tokenize_remove_stopwords': LithuanianRemoveStopwordsTokenizer,
    'lithuanian_lemmatize': LithuanianLemmatizeTokenizer,
    'lithuanian_lemmatize_filter': LithuanianLemmatizeFilterTokenizer,
    'lithuanian_lemmatize_remove_stopwords': LithuanianLemmatizeRemoveStopwordsFilterTokenizer,
    'multi_tokenize': MultiTokenizer,
    'multi_tokenize_filter': MultiFilterTokenizer,
    'multi_tokenize_remove_stopwords': MultiRemoveStopwordsTokenizer,
    'multi_lemmatize': MultiLemmatizeTokenizer,
    'multi_lemmatize_filter': MultiLemmatizeFilterTokenizer,
    'multi_lemmatize_remove_stopwords': MultiLemmatizeRemoveStopwordsTokenizer,
    'bert': BERTTokenizer
}
