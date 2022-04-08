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

import re
from abc import abstractmethod
from typing import List, Union

import torch
import torchtext

from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

SPLIT_REGEX = re.compile(r"\s+")
SPACE_PUNCTUATION_REGEX = re.compile(r"\w+|[^\w\s]")
COMMA_REGEX = re.compile(r"\s*,\s*")
UNDERSCORE_REGEX = re.compile(r"\s*_\s*")


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


# scriptable tokenizers
class SentencePieceTokenizer(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__()

        self.tokenizer = torchtext.transforms.SentencePieceTokenizer(sp_model_path=pretrained_model_name_or_path)

    def forward(self, v: Union[List[str], torch.Tensor]):
        if isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")
        return self.tokenizer(v)


class CLIPTokenizer(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__()

        self.tokenizer = torchtext.transforms.CLIPTokenizer(merges_path=pretrained_model_name_or_path)

    def forward(self, v: Union[List[str], torch.Tensor]):
        if isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")
        return self.tokenizer(v)


tokenizer_registry = {
    "characters": CharactersToListTokenizer,
    "space": SpaceStringToListTokenizer,
    "space_punct": SpacePunctuationStringToListTokenizer,
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
    "hf_tokenizer": HFTokenizer,
    # torchscript-enabled tokenizers
    "sentencepiece_tokenizer": SentencePieceTokenizer,
    "clip_tokenizer": CLIPTokenizer,
}
