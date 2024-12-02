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
from abc import abstractmethod
from typing import Any, List, Union

import torch

from ludwig.utils.hf_utils import load_pretrained_hf_tokenizer
from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text

logger = logging.getLogger(__name__)

TORCHSCRIPT_COMPATIBLE_TOKENIZERS = {"space", "space_punct", "comma", "underscore", "characters"}


class BaseTokenizer:
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, text: str):
        pass

    def convert_token_to_id(self, token: str) -> int:
        raise NotImplementedError()


class StringSplitTokenizer(torch.nn.Module):
    def __init__(self, split_string, **kwargs):
        super().__init__()
        self.split_string = split_string

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
            split_sequence = sequence.strip().split(self.split_string)
            token_sequence: List[str] = []
            for token in self.get_tokens(split_sequence):
                if len(token) > 0:
                    token_sequence.append(token)
            tokens.append(token_sequence)

        return tokens[0] if isinstance(v, str) else tokens

    def get_tokens(self, tokens: List[str]) -> List[str]:
        return tokens


class SpaceStringToListTokenizer(StringSplitTokenizer):
    """Implements torchscript-compatible whitespace tokenization."""

    def __init__(self, **kwargs):
        super().__init__(split_string=" ", **kwargs)


class UnderscoreStringToListTokenizer(StringSplitTokenizer):
    """Implements torchscript-compatible underscore tokenization."""

    def __init__(self, **kwargs):
        super().__init__(split_string="_", **kwargs)


class CommaStringToListTokenizer(StringSplitTokenizer):
    """Implements torchscript-compatible comma tokenization."""

    def __init__(self, **kwargs):
        super().__init__(split_string=",", **kwargs)


class CharactersToListTokenizer(torch.nn.Module):
    """Implements torchscript-compatible characters tokenization."""

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
            split_sequence = [char for char in sequence]
            token_sequence: List[str] = []
            for token in self.get_tokens(split_sequence):
                if len(token) > 0:
                    token_sequence.append(token)
            tokens.append(token_sequence)

        return tokens[0] if isinstance(v, str) else tokens

    def get_tokens(self, tokens: List[str]) -> List[str]:
        return tokens


class NgramTokenizer(SpaceStringToListTokenizer):
    """Implements torchscript-compatible n-gram tokenization."""

    def __init__(self, ngram_size: int = 2, **kwargs):
        super().__init__()
        self.n = ngram_size or 2

    def get_tokens(self, tokens: List[str]) -> List[str]:
        return list(self._ngrams_iterator(tokens, ngrams=self.n))

    def _ngrams_iterator(self, token_list, ngrams):
        """Return an iterator that yields the given tokens and their ngrams. This code is taken from
        https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#ngrams_iterator.

        Args:
            token_list: A list of tokens
            ngrams: the number of ngrams.
        """

        def _get_ngrams(n):
            return zip(*[token_list[i:] for i in range(n)])

        for x in token_list:
            yield x
        for n in range(2, ngrams + 1):
            for x in _get_ngrams(n):
                yield " ".join(x)


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
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = load_pretrained_hf_tokenizer(self.pretrained_model_name_or_path, **kwargs)
        self._set_pad_token()

    def __call__(self, text):
        return self.tokenizer.encode(text, truncation=True)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_pad_token(self) -> str:
        return self.tokenizer.pad_token

    def get_unk_token(self) -> str:
        return self.tokenizer.unk_token

    def _set_pad_token(self) -> None:
        """Sets the pad token and pad token ID for the tokenizer."""

        # CodeGenTokenizer Used by Phi-2
        # GPTNeoXTokenizerFast Used by Pythia
        from transformers import (CodeGenTokenizer, CodeGenTokenizerFast,
                                  CodeLlamaTokenizer, CodeLlamaTokenizerFast,
                                  GPT2Tokenizer, GPT2TokenizerFast,
                                  GPTNeoXTokenizerFast, LlamaTokenizer,
                                  LlamaTokenizerFast)

        # Tokenizers might have the pad token id attribute since they tend to use the same base class, but
        # it can be set to None so we check for this explicitly.
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            return

        # HACK(geoffrey): gpt2 has no pad token. Recommendation is to use eos token instead.
        # https://github.com/huggingface/transformers/issues/2630#issuecomment-1290809338
        # https://github.com/huggingface/transformers/issues/2648#issuecomment-616177044
        if any(
            isinstance(self.tokenizer, t)
            for t in [
                CodeGenTokenizer,
                CodeGenTokenizerFast,
                CodeLlamaTokenizer,
                CodeLlamaTokenizerFast,
                GPT2Tokenizer,
                GPT2TokenizerFast,
                GPTNeoXTokenizerFast,
                LlamaTokenizer,
                LlamaTokenizerFast,
            ]
        ):
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                logger.warning("No padding token id found. Using eos_token as pad_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Incase any HF tokenizer does not have pad token ID, just default to using 0
        # as the pad_token_id.
        if self.tokenizer.pad_token_id is None:
            logger.warning("No padding token id found. Using 0 as pad token id.")
            self.tokenizer.pad_token_id = 0

    def convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)


tokenizer_registry = {
    # Torchscript-compatible tokenizers.
    "space": SpaceStringToListTokenizer,
    "space_punct": SpacePunctuationStringToListTokenizer,
    "ngram": NgramTokenizer,
    "characters": CharactersToListTokenizer,
    "underscore": UnderscoreStringToListTokenizer,
    "comma": CommaStringToListTokenizer,
    # Tokenizers not compatible with torchscript
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


class HFTokenizerShortcutFactory:
    """This factory can be used to build HuggingFace tokenizers form a shortcut string.

    Those shortcuts were originally used for torchtext tokenizers. They also guarantee backward compatibility.
    """

    MODELS = {
        "sentencepiece": "FacebookAI/xlm-roberta-base",
        "clip": "openai/clip-vit-base-patch32",
        "gpt2bpe": "openai-community/gpt2",
        "bert": "bert-base-uncased",
    }

    @classmethod
    def create_class(cls, model_name: str):
        """Creating a tokenizer class from a model name."""

        class DynamicHFTokenizer(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.tokenizer = load_pretrained_hf_tokenizer(model_name, use_fast=False)

            def forward(self, v: Union[str, List[str], torch.Tensor]):
                if isinstance(v, torch.Tensor):
                    raise ValueError(f"Unsupported input: {v}")
                return self.tokenizer.tokenize(v)

        return DynamicHFTokenizer


tokenizer_registry.update(
    {name: HFTokenizerShortcutFactory.create_class(model) for name, model in HFTokenizerShortcutFactory.MODELS.items()}
)


def get_hf_tokenizer(pretrained_model_name_or_path, **kwargs):
    """Gets a potentially torchscript-compatible tokenizer that follows HF convention.

    Args:
        pretrained_model_name_or_path: Name of the model in the HF repo. Example: "bert-base-uncased".
    Returns:
        A torchscript-able HF tokenizer if it is available. Else, returns vanilla HF tokenizer.
    """

    return HFTokenizer(pretrained_model_name_or_path)


tokenizer_registry.update(
    {
        "hf_tokenizer": get_hf_tokenizer,
    }
)


def get_tokenizer_from_registry(tokenizer_name: str) -> torch.nn.Module:
    """Returns the appropriate tokenizer from the tokenizer registry.

    Raises a KeyError if a tokenizer that does not exist in the registry is requested, with additional help text if the
    requested tokenizer would be available for a different version of torchtext.
    """
    if tokenizer_name in tokenizer_registry:
        return tokenizer_registry[tokenizer_name]
    # Tokenizer does not exist or is unavailable.
    raise KeyError(f"Invalid tokenizer name: '{tokenizer_name}'. Available tokenizers: {tokenizer_registry.keys()}")
