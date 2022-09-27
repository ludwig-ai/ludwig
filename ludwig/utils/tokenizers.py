"""Ludwig string tokenizers including string-based, spacy-based, and huggingface-based implementations.

To add a new tokenizer, 1) implement a subclass of BaseTokenizer and 2) add it to the tokenizer_registry.

Once it's in the registry, tokenizers can be used in a ludwig config, e.g..

```
input_features:
    -   name: title
        type: text
        preprocessing:
            tokenizer: <NEW_TOKENIZER>
```
"""

import logging
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from ludwig.utils.data_utils import load_json
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

logger = logging.getLogger(__name__)


SPACE_PUNCTUATION_REGEX = re.compile(r"\w+|[^\w\s]")
COMMA_REGEX = re.compile(r"\s*,\s*")
UNDERSCORE_REGEX = re.compile(r"\s*_\s*")

TORCHSCRIPT_COMPATIBLE_TOKENIZERS = {"space", "space_punct"}
TORCHTEXT_0_12_0_TOKENIZERS = {"sentencepiece", "clip", "gpt2bpe"}
TORCHTEXT_0_13_0_TOKENIZERS = {"bert"}

# Do not use torchtext implementation of BERT tokenizer for these model names:
# https://github.com/pytorch/text/issues/1840
SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES = {
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "TurkuNLP/bert-base-finnish-cased-v1",
}


class BaseTokenizer:
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, text: str):
        pass


class CharactersToListTokenizer(BaseTokenizer):
    def __call__(self, text):
        return [char for char in text]


class SpaceStringToListTokenizer(torch.nn.Module):
    """Implements torchscript-compatible whitespace tokenization."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, v: Union[str, List[str], torch.Tensor]) -> Any:
        if isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")

        inputs: List[str] = []
        # Ludwig calls map on List[str] objects, so we need to handle individual strings as well.
        if isinstance(v, str):
            inputs.append(v)
        else:
            inputs.extend(v)

        tokens: List[List[str]] = []
        for sequence in inputs:
            split_sequence = sequence.strip().split(" ")
            token_sequence: List[str] = []
            for token in split_sequence:
                if len(token) > 0:
                    token_sequence.append(token)
            tokens.append(token_sequence)

        return tokens[0] if isinstance(v, str) else tokens


class SpacePunctuationStringToListTokenizer(torch.nn.Module):
    """Implements torchscript-compatible space_punct tokenization."""

    def __init__(self, **kwargs):
        super().__init__()

    def is_regex_w(self, c: str) -> bool:
        return c.isalnum() or c == "_"

    def forward(self, v: Union[str, List[str], torch.Tensor]) -> Any:
        if isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")

        inputs: List[str] = []
        # Ludwig calls map on List[str] objects, so we need to handle individual strings as well.
        if isinstance(v, str):
            inputs.append(v)
        else:
            inputs.extend(v)

        tokens: List[List[str]] = []
        for sequence in inputs:
            token_sequence: List[str] = []
            word: List[str] = []
            for c in sequence:
                if self.is_regex_w(c):
                    word.append(c)
                elif len(word) > 0:  # if non-empty word and non-alphanumeric char, append word to token sequence
                    token_sequence.append("".join(word))
                    word.clear()

                if not self.is_regex_w(c) and not c.isspace():  # non-alphanumeric, non-space char is punctuation
                    token_sequence.append(c)

            if len(word) > 0:  # add last word
                token_sequence.append("".join(word))

            tokens.append(token_sequence)

        return tokens[0] if isinstance(v, str) else tokens


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
        return process_text(text, load_nlp_pipeline("en"))


class EnglishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("en"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class EnglishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("en"), filter_stopwords=True)


class EnglishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        process_text(text, load_nlp_pipeline("en"), return_lemma=True)


class EnglishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("en"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class EnglishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("en"), return_lemma=True, filter_stopwords=True)


class ItalianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("it"))


class ItalianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("it"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class ItalianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("it"), filter_stopwords=True)


class ItalianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("it"), return_lemma=True)


class ItalianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("it"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class ItalianLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("it"), return_lemma=True, filter_stopwords=True)


class SpanishTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("es"))


class SpanishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("es"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class SpanishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("es"), filter_stopwords=True)


class SpanishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("es"), return_lemma=True)


class SpanishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("es"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class SpanishLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("es"), return_lemma=True, filter_stopwords=True)


class GermanTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("de"))


class GermanFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("de"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class GermanRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("de"), filter_stopwords=True)


class GermanLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("de"), return_lemma=True)


class GermanLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("de"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class GermanLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("de"), return_lemma=True, filter_stopwords=True)


class FrenchTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("fr"))


class FrenchFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("fr"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class FrenchRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("fr"), filter_stopwords=True)


class FrenchLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("fr"), return_lemma=True)


class FrenchLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("fr"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class FrenchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("fr"), return_lemma=True, filter_stopwords=True)


class PortugueseTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pt"))


class PortugueseFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("pt"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class PortugueseRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pt"), filter_stopwords=True)


class PortugueseLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pt"), return_lemma=True)


class PortugueseLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("pt"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class PortugueseLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pt"), return_lemma=True, filter_stopwords=True)


class DutchTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nl"))


class DutchFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("nl"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class DutchRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nl"), filter_stopwords=True)


class DutchLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nl"), return_lemma=True)


class DutchLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("nl"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class DutchLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nl"), return_lemma=True, filter_stopwords=True)


class GreekTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("el"))


class GreekFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("el"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class GreekRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("el"), filter_stopwords=True)


class GreekLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("el"), return_lemma=True)


class GreekLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("el"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class GreekLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("el"), return_lemma=True, filter_stopwords=True)


class NorwegianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nb"))


class NorwegianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("nb"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class NorwegianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nb"), filter_stopwords=True)


class NorwegianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nb"), return_lemma=True)


class NorwegianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("nb"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class NorwegianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("nb"), return_lemma=True, filter_stopwords=True)


class LithuanianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("lt"))


class LithuanianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("lt"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class LithuanianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("lt"), filter_stopwords=True)


class LithuanianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("lt"), return_lemma=True)


class LithuanianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("lt"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class LithuanianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("lt"), return_lemma=True, filter_stopwords=True)


class DanishTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("da"))


class DanishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("da"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class DanishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("da"), filter_stopwords=True)


class DanishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("da"), return_lemma=True)


class DanishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("da"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class DanishLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("da"), return_lemma=True, filter_stopwords=True)


class PolishTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pl"))


class PolishFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("pl"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class PolishRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pl"), filter_stopwords=True)


class PolishLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pl"), return_lemma=True)


class PolishLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("pl"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class PolishLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("pl"), return_lemma=True, filter_stopwords=True)


class RomanianTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("ro"))


class RomanianFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("ro"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class RomanianRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("ro"), filter_stopwords=True)


class RomanianLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("ro"), return_lemma=True)


class RomanianLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("ro"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class RomanianLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("ro"), return_lemma=True, filter_stopwords=True)


class JapaneseTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("jp"))


class JapaneseFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("jp"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class JapaneseRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("jp"), filter_stopwords=True)


class JapaneseLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("jp"), return_lemma=True)


class JapaneseLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("jp"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class JapaneseLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("jp"), return_lemma=True, filter_stopwords=True)


class ChineseTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("zh"))


class ChineseFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("zh"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class ChineseRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("zh"), filter_stopwords=True)


class ChineseLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("zh"), return_lemma=True)


class ChineseLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("zh"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class ChineseLemmatizeRemoveStopwordsFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("zh"), return_lemma=True, filter_stopwords=True)


class MultiTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("xx"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class MultiFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text, load_nlp_pipeline("xx"), filter_numbers=True, filter_punctuation=True, filter_short_tokens=True
        )


class MultiRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("xx"), filter_stopwords=True)


class MultiLemmatizeTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("xx"), return_lemma=True)


class MultiLemmatizeFilterTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(
            text,
            load_nlp_pipeline("xx"),
            return_lemma=True,
            filter_numbers=True,
            filter_punctuation=True,
            filter_short_tokens=True,
        )


class MultiLemmatizeRemoveStopwordsTokenizer(BaseTokenizer):
    def __call__(self, text):
        return process_text(text, load_nlp_pipeline("xx"), return_lemma=True, filter_stopwords=True)


class HFTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
        )

    def __call__(self, text):
        return self.tokenizer.encode(text, truncation=True)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_pad_token(self) -> str:
        return self.tokenizer.pad_token

    def get_unk_token(self) -> str:
        return self.tokenizer.unk_token


tokenizer_registry = {
    # Torchscript-compatible tokenizers. Torchtext tokenizers are also available below (requires torchtext>=0.12.0).
    "space": SpaceStringToListTokenizer,
    "space_punct": SpacePunctuationStringToListTokenizer,
    # Tokenizers not compatible with torchscript
    "characters": CharactersToListTokenizer,
    "underscore": UnderscoreStringToListTokenizer,
    "comma": CommaStringToListTokenizer,
    "untokenized": UntokenizedStringToListTokenizer,
    "stripped": StrippedStringToListTokenizer,
    "english_tokenize": EnglishTokenizer,
    "english_tokenize_filter": EnglishFilterTokenizer,
    "english_tokenize_remove_stopwords": EnglishRemoveStopwordsTokenizer,
    "english_lemmatize": EnglishLemmatizeTokenizer,
    "english_lemmatize_filter": EnglishLemmatizeFilterTokenizer,
    "english_lemmatize_remove_stopwords": EnglishLemmatizeRemoveStopwordsTokenizer,
    "italian_tokenize": ItalianTokenizer,
    "italian_tokenize_filter": ItalianFilterTokenizer,
    "italian_tokenize_remove_stopwords": ItalianRemoveStopwordsTokenizer,
    "italian_lemmatize": ItalianLemmatizeTokenizer,
    "italian_lemmatize_filter": ItalianLemmatizeFilterTokenizer,
    "italian_lemmatize_remove_stopwords": ItalianLemmatizeRemoveStopwordsTokenizer,
    "spanish_tokenize": SpanishTokenizer,
    "spanish_tokenize_filter": SpanishFilterTokenizer,
    "spanish_tokenize_remove_stopwords": SpanishRemoveStopwordsTokenizer,
    "spanish_lemmatize": SpanishLemmatizeTokenizer,
    "spanish_lemmatize_filter": SpanishLemmatizeFilterTokenizer,
    "spanish_lemmatize_remove_stopwords": SpanishLemmatizeRemoveStopwordsTokenizer,
    "german_tokenize": GermanTokenizer,
    "german_tokenize_filter": GermanFilterTokenizer,
    "german_tokenize_remove_stopwords": GermanRemoveStopwordsTokenizer,
    "german_lemmatize": GermanLemmatizeTokenizer,
    "german_lemmatize_filter": GermanLemmatizeFilterTokenizer,
    "german_lemmatize_remove_stopwords": GermanLemmatizeRemoveStopwordsTokenizer,
    "french_tokenize": FrenchTokenizer,
    "french_tokenize_filter": FrenchFilterTokenizer,
    "french_tokenize_remove_stopwords": FrenchRemoveStopwordsTokenizer,
    "french_lemmatize": FrenchLemmatizeTokenizer,
    "french_lemmatize_filter": FrenchLemmatizeFilterTokenizer,
    "french_lemmatize_remove_stopwords": FrenchLemmatizeRemoveStopwordsTokenizer,
    "portuguese_tokenize": PortugueseTokenizer,
    "portuguese_tokenize_filter": PortugueseFilterTokenizer,
    "portuguese_tokenize_remove_stopwords": PortugueseRemoveStopwordsTokenizer,
    "portuguese_lemmatize": PortugueseLemmatizeTokenizer,
    "portuguese_lemmatize_filter": PortugueseLemmatizeFilterTokenizer,
    "portuguese_lemmatize_remove_stopwords": PortugueseLemmatizeRemoveStopwordsTokenizer,
    "dutch_tokenize": DutchTokenizer,
    "dutch_tokenize_filter": DutchFilterTokenizer,
    "dutch_tokenize_remove_stopwords": DutchRemoveStopwordsTokenizer,
    "dutch_lemmatize": DutchLemmatizeTokenizer,
    "dutch_lemmatize_filter": DutchLemmatizeFilterTokenizer,
    "dutch_lemmatize_remove_stopwords": DutchLemmatizeRemoveStopwordsTokenizer,
    "greek_tokenize": GreekTokenizer,
    "greek_tokenize_filter": GreekFilterTokenizer,
    "greek_tokenize_remove_stopwords": GreekRemoveStopwordsTokenizer,
    "greek_lemmatize": GreekLemmatizeTokenizer,
    "greek_lemmatize_filter": GreekLemmatizeFilterTokenizer,
    "greek_lemmatize_remove_stopwords": GreekLemmatizeRemoveStopwordsFilterTokenizer,
    "norwegian_tokenize": NorwegianTokenizer,
    "norwegian_tokenize_filter": NorwegianFilterTokenizer,
    "norwegian_tokenize_remove_stopwords": NorwegianRemoveStopwordsTokenizer,
    "norwegian_lemmatize": NorwegianLemmatizeTokenizer,
    "norwegian_lemmatize_filter": NorwegianLemmatizeFilterTokenizer,
    "norwegian_lemmatize_remove_stopwords": NorwegianLemmatizeRemoveStopwordsFilterTokenizer,
    "lithuanian_tokenize": LithuanianTokenizer,
    "lithuanian_tokenize_filter": LithuanianFilterTokenizer,
    "lithuanian_tokenize_remove_stopwords": LithuanianRemoveStopwordsTokenizer,
    "lithuanian_lemmatize": LithuanianLemmatizeTokenizer,
    "lithuanian_lemmatize_filter": LithuanianLemmatizeFilterTokenizer,
    "lithuanian_lemmatize_remove_stopwords": LithuanianLemmatizeRemoveStopwordsFilterTokenizer,
    "danish_tokenize": DanishTokenizer,
    "danish_tokenize_filter": DanishFilterTokenizer,
    "danish_tokenize_remove_stopwords": DanishRemoveStopwordsTokenizer,
    "danish_lemmatize": DanishLemmatizeTokenizer,
    "danish_lemmatize_filter": DanishLemmatizeFilterTokenizer,
    "danish_lemmatize_remove_stopwords": DanishLemmatizeRemoveStopwordsFilterTokenizer,
    "polish_tokenize": PolishTokenizer,
    "polish_tokenize_filter": PolishFilterTokenizer,
    "polish_tokenize_remove_stopwords": PolishRemoveStopwordsTokenizer,
    "polish_lemmatize": PolishLemmatizeTokenizer,
    "polish_lemmatize_filter": PolishLemmatizeFilterTokenizer,
    "polish_lemmatize_remove_stopwords": PolishLemmatizeRemoveStopwordsFilterTokenizer,
    "romanian_tokenize": RomanianTokenizer,
    "romanian_tokenize_filter": RomanianFilterTokenizer,
    "romanian_tokenize_remove_stopwords": RomanianRemoveStopwordsTokenizer,
    "romanian_lemmatize": RomanianLemmatizeTokenizer,
    "romanian_lemmatize_filter": RomanianLemmatizeFilterTokenizer,
    "romanian_lemmatize_remove_stopwords": RomanianLemmatizeRemoveStopwordsFilterTokenizer,
    "japanese_tokenize": JapaneseTokenizer,
    "japanese_tokenize_filter": JapaneseFilterTokenizer,
    "japanese_tokenize_remove_stopwords": JapaneseRemoveStopwordsTokenizer,
    "japanese_lemmatize": JapaneseLemmatizeTokenizer,
    "japanese_lemmatize_filter": JapaneseLemmatizeFilterTokenizer,
    "japanese_lemmatize_remove_stopwords": JapaneseLemmatizeRemoveStopwordsFilterTokenizer,
    "chinese_tokenize": ChineseTokenizer,
    "chinese_tokenize_filter": ChineseFilterTokenizer,
    "chinese_tokenize_remove_stopwords": ChineseRemoveStopwordsTokenizer,
    "chinese_lemmatize": ChineseLemmatizeTokenizer,
    "chinese_lemmatize_filter": ChineseLemmatizeFilterTokenizer,
    "chinese_lemmatize_remove_stopwords": ChineseLemmatizeRemoveStopwordsFilterTokenizer,
    "multi_tokenize": MultiTokenizer,
    "multi_tokenize_filter": MultiFilterTokenizer,
    "multi_tokenize_remove_stopwords": MultiRemoveStopwordsTokenizer,
    "multi_lemmatize": MultiLemmatizeTokenizer,
    "multi_lemmatize_filter": MultiLemmatizeFilterTokenizer,
    "multi_lemmatize_remove_stopwords": MultiLemmatizeRemoveStopwordsTokenizer,
}

try:
    import torchtext

    if torch.torch_version.TorchVersion(torchtext.__version__) >= (0, 12, 0):
        """torchtext 0.12.0 tokenizers.

        Only available with torchtext>=0.12.0.
        """

        class SentencePieceTokenizer(torch.nn.Module):
            def __init__(self, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
                super().__init__()
                if pretrained_model_name_or_path is None:
                    pretrained_model_name_or_path = (
                        "https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
                    )
                self.tokenizer = torchtext.transforms.SentencePieceTokenizer(
                    sp_model_path=pretrained_model_name_or_path
                )

            def forward(self, v: Union[str, List[str], torch.Tensor]):
                if isinstance(v, torch.Tensor):
                    raise ValueError(f"Unsupported input: {v}")
                return self.tokenizer(v)

        class _BPETokenizer(torch.nn.Module):
            """Superclass for tokenizers that use BPE, such as CLIPTokenizer and GPT2BPETokenizer."""

            def __init__(self, pretrained_model_name_or_path: str, vocab_file: str):
                super().__init__()
                self.str2idx, self.idx2str = self._init_vocab(vocab_file)
                # TODO(geoffrey): If we move to torchtext>=0.13.0, we can use return_tokens kwarg to get tokens directly
                self.tokenizer = self._init_tokenizer(pretrained_model_name_or_path, vocab_file)

            def _init_vocab(self, vocab_file: str) -> Dict[str, str]:
                """Loads the vocab from the vocab file."""
                str2idx = load_json(torchtext.utils.get_asset_local_path(vocab_file))
                _, idx2str = zip(*sorted((v, k) for k, v in str2idx.items()))
                return str2idx, idx2str

            def _init_tokenizer(self, pretrained_model_name_or_path: str, vocab_file: str) -> Any:
                """Initializes and returns the tokenizer."""
                raise NotImplementedError

            def forward(self, v: Union[str, List[str], torch.Tensor]) -> Any:
                """Implements forward pass for tokenizer.

                BPE tokenizers from torchtext return ids directly, which is inconsistent with the Ludwig tokenizer API.
                The below implementation works around this by converting the ids back to their original string tokens.
                """
                if isinstance(v, torch.Tensor):
                    raise ValueError(f"Unsupported input: {v}")

                inputs: List[str] = []
                # Ludwig calls map on List[str] objects, so we need to handle individual strings as well.
                if isinstance(v, str):
                    inputs.append(v)
                else:
                    inputs.extend(v)

                token_ids = self.tokenizer(inputs)
                assert torch.jit.isinstance(token_ids, List[List[str]])

                tokens = [[self.idx2str[int(unit_idx)] for unit_idx in sequence] for sequence in token_ids]
                return tokens[0] if isinstance(v, str) else tokens

            def get_vocab(self) -> Dict[str, str]:
                return self.str2idx

        class CLIPTokenizer(_BPETokenizer):
            def __init__(
                self, pretrained_model_name_or_path: Optional[str] = None, vocab_file: Optional[str] = None, **kwargs
            ):
                if pretrained_model_name_or_path is None:
                    pretrained_model_name_or_path = "http://download.pytorch.org/models/text/clip_merges.bpe"
                if vocab_file is None:
                    vocab_file = "http://download.pytorch.org/models/text/clip_encoder.json"
                super().__init__(pretrained_model_name_or_path, vocab_file)

            def _init_tokenizer(self, pretrained_model_name_or_path: str, vocab_file: str):
                return torchtext.transforms.CLIPTokenizer(
                    encoder_json_path=vocab_file, merges_path=pretrained_model_name_or_path
                )

        class GPT2BPETokenizer(_BPETokenizer):
            def __init__(
                self, pretrained_model_name_or_path: Optional[str] = None, vocab_file: Optional[str] = None, **kwargs
            ):
                if pretrained_model_name_or_path is None:
                    pretrained_model_name_or_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
                if vocab_file is None:
                    vocab_file = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
                super().__init__(pretrained_model_name_or_path, vocab_file)

            def _init_tokenizer(self, pretrained_model_name_or_path: str, vocab_file: str):
                return torchtext.transforms.GPT2BPETokenizer(
                    encoder_json_path=vocab_file, vocab_bpe_path=pretrained_model_name_or_path
                )

        tokenizer_registry.update(
            {
                "sentencepiece": SentencePieceTokenizer,
                "clip": CLIPTokenizer,
                "gpt2bpe": GPT2BPETokenizer,
            }
        )
        TORCHSCRIPT_COMPATIBLE_TOKENIZERS.update(TORCHTEXT_0_12_0_TOKENIZERS)
    else:
        raise ImportError

except ImportError:
    logger.warning(
        f"torchtext>=0.12.0 is not installed, so the following tokenizers are not available: "
        f"{TORCHTEXT_0_12_0_TOKENIZERS}"
    )


try:
    import torchtext

    if torch.torch_version.TorchVersion(torchtext.__version__) >= (0, 13, 0):
        pass
    else:
        raise ImportError

    class BERTTokenizer(torch.nn.Module):
        def __init__(
            self,
            vocab_file: Optional[str] = None,
            is_hf_tokenizer: Optional[bool] = False,
            do_lower_case: Optional[bool] = False,
            **kwargs,
        ):
            super().__init__()
            from transformers.utils.hub import cached_path

            if vocab_file is None:
                # By default, we use the uncased (all lowercased) BERT tokenizer from huggingface.
                vocab_file = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
                do_lower_case = True
            self.vocab_file = cached_path(vocab_file)

            self.is_hf_tokenizer = is_hf_tokenizer

            # Return tokens as raw tokens only if not being used as a HF tokenizer.
            self.return_tokens = not self.is_hf_tokenizer
            self.tokenizer = torchtext.transforms.BERTTokenizer(
                vocab_path=self.vocab_file, return_tokens=self.return_tokens, do_lower_case=do_lower_case
            )

            self.str2idx = self._init_vocab(self.vocab_file)

            # Values from
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
            self.pad_token = "[PAD]"
            self.unk_token = "[UNK]"
            self.cls_token_id = self.str2idx["[CLS]"]  # Used as start symbol
            self.sep_token_id = self.str2idx["[SEP]"]  # Used as stop symbol

        def _init_vocab(self, vocab_file: str) -> Dict[str, str]:
            str2idx = {}
            with open(vocab_file, encoding="utf-8") as reader:
                tokens = reader.readlines()
            for index, token in enumerate(tokens):
                token = token.rstrip("\n")
                str2idx[token] = index
            return str2idx

        def forward(self, v: Union[str, List[str], torch.Tensor]) -> Any:
            """Implements forward pass for tokenizer.

            If the is_hf_tokenizer flag is set to True, then the output follows the HF convention, i.e. the output is an
            List[List[int]] of tokens and the cls and sep tokens are automatically added as the start and stop symbols.

            If the is_hf_tokenizer flag is set to False, then the output follows the Ludwig convention, i.e. the output
            is a List[List[str]] of tokens.
            """
            if isinstance(v, torch.Tensor):
                raise ValueError(f"Unsupported input: {v}")

            inputs: List[str] = []
            # Ludwig calls map on List[str] objects, so we need to handle individual strings as well.
            if isinstance(v, str):
                inputs.append(v)
            else:
                inputs.extend(v)

            if self.is_hf_tokenizer:
                token_ids_str = self.tokenizer(inputs)
                assert torch.jit.isinstance(token_ids_str, List[List[str]])
                # Must cast token_ids to ints because they are used directly as indices.
                token_ids: List[List[int]] = []
                for token_ids_str_i in token_ids_str:
                    token_ids_i = [int(token_id_str) for token_id_str in token_ids_str_i]
                    token_ids_i.insert(0, self.cls_token_id)
                    token_ids_i.append(self.sep_token_id)
                    token_ids.append(token_ids_i)
                return token_ids[0] if isinstance(v, str) else token_ids

            tokens = self.tokenizer(inputs)
            assert torch.jit.isinstance(tokens, List[List[str]])
            return tokens[0] if isinstance(v, str) else tokens

        def get_vocab(self) -> Dict[str, str]:
            return self.str2idx

        def get_pad_token(self) -> str:
            return self.pad_token

        def get_unk_token(self) -> str:
            return self.unk_token

    tokenizer_registry.update(
        {
            "bert": BERTTokenizer,
        }
    )
    TORCHSCRIPT_COMPATIBLE_TOKENIZERS.update(TORCHTEXT_0_13_0_TOKENIZERS)

except ImportError:
    logger.warning(
        f"torchtext>=0.13.0 is not installed, so the following tokenizers are not available: "
        f"{TORCHTEXT_0_13_0_TOKENIZERS}"
    )


def get_hf_tokenizer(pretrained_model_name_or_path, **kwargs):
    """Gets a potentially torchscript-compatible tokenizer that follows HF convention.

    Args:
        pretrained_model_name_or_path: Name of the model in the HF repo. Example: "bert-base-uncased".
    Returns:
        A torchscript-able HF tokenizer if it is available. Else, returns vanilla HF tokenizer.
    """

    if "bert" in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
        from transformers.models.bert.tokenization_bert import PRETRAINED_INIT_CONFIGURATION, PRETRAINED_VOCAB_FILES_MAP

        if (
            pretrained_model_name_or_path in PRETRAINED_VOCAB_FILES_MAP["vocab_file"]
            and pretrained_model_name_or_path not in SKIP_TORCHTEXT_BERT_HF_MODEL_NAMES
        ):
            logger.info(f"Loading TorchText implementation of {pretrained_model_name_or_path} tokenizer")
            vocab_file = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][pretrained_model_name_or_path]
            init_kwargs = PRETRAINED_INIT_CONFIGURATION.get(pretrained_model_name_or_path, {})
            return BERTTokenizer(
                vocab_file,
                is_hf_tokenizer=True,
                **init_kwargs,
            )

    # If pretrained_model_name_or_path does not have a torchtext equivalent implementation, load the
    # HuggingFace implementation.
    logger.info(f"Loading HuggingFace implementation of {pretrained_model_name_or_path} tokenizer")
    return HFTokenizer(pretrained_model_name_or_path)


tokenizer_registry.update(
    {
        "hf_tokenizer": get_hf_tokenizer,
    }
)
