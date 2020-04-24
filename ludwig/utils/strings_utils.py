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
import re
import unicodedata
from abc import abstractmethod
from collections import Counter

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

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
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocabulary = []
        for line in f:
            line = line.strip()
            if ' ' in line:
                line = line.split(' ')[0]
            vocabulary.append(line)
        return vocabulary
        # return [line.strip() for line in f]


def create_vocabulary(
        data,
        tokenizer_type='space',
        add_unknown=True,
        add_padding=True,
        lowercase=True,
        num_most_frequent=None,
        vocab_file=None,
        unknown_symbol=UNKNOWN_SYMBOL,
        padding_symbol=PADDING_SYMBOL,
        pretrained_model_name_or_path=None
):
    vocab = None
    max_line_length = 0
    unit_counts = Counter()

    tokenizer = get_from_registry(
        tokenizer_type,
        tokenizer_registry
    )(
        vocab_file=vocab_file,
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )

    if tokenizer_type == 'bert':
        vocab = load_vocabulary(vocab_file)
        add_unknown = False
        add_padding = False
    elif tokenizer_type == 'hf_tokenizer':
        vocab = tokenizer.tokenizer.get_vocab()
        add_unknown = False
        add_padding = False
    elif vocab_file is not None:
        vocab = load_vocabulary(vocab_file)

    for line in data:
        processed_line = tokenizer.tokenize(line.lower() if lowercase else line)
        max_line_length = max(max_line_length, len(processed_line))
        if vocab is None:
            unit_counts.update(processed_line)

    if vocab is None:
        vocab = [unit for unit, count in
                 unit_counts.most_common(num_most_frequent)]

    if add_unknown or add_padding:
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


def build_sequence_matrix(
        sequences,
        max_length,
        vocab_size,
        padding='pre',
):
    sequence_matrix = pad_sequences(
        sequences,
        maxlen=max_length,
        dtype=int_type(vocab_size - 1),
        padding=padding,
        truncating=padding,
        value=0.0
    )
    return sequence_matrix


def get_tokenizer(
        tokenizer_type='space',
        lowercase=True,
        add_unknown=True,
        add_padding=True,
        vocab=None,
        symbols=None,
        max_length=None,
        vocab_file=None,
        pretrained_model_name_or_path=None
):
    tokenizer = get_from_registry(
        tokenizer_type,
        tokenizer_registry
    )(
        lowercase=lowercase,
        add_unknown=add_unknown,
        add_padding=add_padding,
        vocab=vocab,
        symbols=symbols,
        max_length=max_length,
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    if vocab_file:
        tokenizer.load_covabulary(vocab_file)
    return tokenizer


class BaseTokenizer:

    def __init__(
            self,
            lowercase=True,
            add_unknown_symbol=True,
            add_padding_symbol=True,
            vocab=None,
            symbols=None,
            max_length=None,
            **kwargs
    ):
        self.lowercase = lowercase
        self.add_unknown_symbol = add_unknown_symbol
        self.add_padding_symbol = add_padding_symbol
        self.vocab = vocab
        self.symbols = symbols
        self.max_length = max_length

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, token_sequence):
        pass

    @abstractmethod
    def tokenize_id(self, text):
        return [self.vocab[token] for token in self.tokenize(text)]

    @abstractmethod
    def detokenize_id(self, id_sequence):
        pass

    def load_vocabulary(self, vocab_file):
        self.vocab = load_vocabulary(vocab_file)

    def fit_vocab(
            self,
            data,
            num_most_frequent=20000,
    ):
        max_length = 0
        unit_counts = Counter()

        for line in data:
            processed_line = self.tokenize(line)
            max_length = max(max_length, len(processed_line))
            unit_counts.update(processed_line)

        symbols = [unit for unit, count in
                   unit_counts.most_common(num_most_frequent)]

        if self.add_unknown_symbol or self.add_padding_symbol:
            symbols_set = set(symbols)
            if self.add_unknown_symbol:
                if UNKNOWN_SYMBOL not in symbols_set:
                    symbols = [UNKNOWN_SYMBOL] + symbols
            if self.add_padding_symbol:
                if PADDING_SYMBOL not in symbols_set:
                    symbols = [PADDING_SYMBOL] + symbols

        self.vocab = {unit: i for i, unit in enumerate(symbols)}
        self.symbols = symbols
        self.max_length = max_length

    def fit_max_length(
            self,
            data,
    ):
        max_length = 0
        for line in data:
            processed_line = self.tokenize(line)
            max_length = max(max_length, len(processed_line))
        self.max_length = max_length

class CharactersToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return text


class SpaceStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return SPLIT_REGEX.split(text.strip())


class SpacePunctuationStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return SPACE_PUNCTUATION_REGEX.findall(text.strip())


class UnderscoreStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return UNDERSCORE_REGEX.split(text.strip())


class CommaStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return COMMA_REGEX.split(text.strip())


class UntokenizedStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return [text]


class StrippedStringToListTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return [text.strip()]


class EnglishTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('en'))


class EnglishFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class EnglishRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            filter_stopwords=True
        )


class EnglishLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        process_text(text, load_nlp_pipeline('en'), return_lemma=True)


class EnglishLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class EnglishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('en'),
            return_lemma=True,
            filter_stopwords=True
        )


class ItalianTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('it'))


class ItalianFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class ItalianRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            filter_stopwords=True
        )


class ItalianLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True
        )


class ItalianLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class ItalianLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('it'),
            return_lemma=True,
            filter_stopwords=True
        )


class SpanishTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('es'))


class SpanishFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class SpanishRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            filter_stopwords=True
        )


class SpanishLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True
        )


class SpanishLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class SpanishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('es'),
            return_lemma=True,
            filter_stopwords=True
        )


class GermanTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('de'))


class GermanFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GermanRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            filter_stopwords=True
        )


class GermanLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True
        )


class GermanLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GermanLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('de'),
            return_lemma=True,
            filter_stopwords=True
        )


class FrenchTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('fr'))


class FrenchFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class FrenchRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            filter_stopwords=True
        )


class FrenchLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True
        )


class FrenchLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class FrenchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('fr'),
            return_lemma=True,
            filter_stopwords=True
        )


class PortugueseTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('pt'))


class PortugueseFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class PortugueseRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            filter_stopwords=True
        )


class PortugueseLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('pt'), return_lemma=True)


class PortugueseLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class PortugueseLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('pt'),
            return_lemma=True,
            filter_stopwords=True
        )


class DutchTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('nl'))


class DutchFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class DutchRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            filter_stopwords=True
        )


class DutchLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('nl'), return_lemma=True)


class DutchLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class DutchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nl'),
            return_lemma=True,
            filter_stopwords=True
        )


class GreekTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('el'))


class GreekFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GreekRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            filter_stopwords=True
        )


class GreekLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('el'), return_lemma=True)


class GreekLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class GreekLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('el'),
            return_lemma=True,
            filter_stopwords=True
        )


class NorwegianTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('nb'))


class NorwegianFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class NorwegianRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            filter_stopwords=True
        )


class NorwegianLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('nb'), return_lemma=True)


class NorwegianLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class NorwegianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('nb'),
            return_lemma=True,
            filter_stopwords=True
        )


class LithuanianTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('lt'))


class LithuanianFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class LithuanianRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            filter_stopwords=True
        )


class LithuanianLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('lt'), return_lemma=True)


class LithuanianLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class LithuanianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('lt'),
            return_lemma=True,
            filter_stopwords=True
        )


class MultiTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('xx'))


class MultiFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class MultiRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            filter_stopwords=True
        )


class MultiLemmatizeTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(text, load_nlp_pipeline('xx'), return_lemma=True)


class MultiLemmatizeFilterTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True
        )


class MultiLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def tokenize(self, text):
        return process_text(
            text,
            load_nlp_pipeline('xx'),
            return_lemma=True,
            filter_stopwords=True
        )


class HFTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.vocab = self.tokenizer.vocab
        self.max_model_input_size = self.tokenizer.max_model_input_sizes[
            pretrained_model_name_or_path
        ]

    def fit_max_length(
            self,
            data,
    ):
        super().fit_max_length(data)
        self.max_length = min(self.max_length, self.max_model_input_size)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def detokenize(self, token_sequence):
        return self.tokenizer.convert_tokens_to_string(token_sequence)

    def tokenize_id(self, text):
        return self.tokenizer.encode(text)

    def detokenize_id(self, id_sequence):
        return self.tokenizer.decode(id_sequence)


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
    'hf_tokenizer': HFTokenizer,
}
