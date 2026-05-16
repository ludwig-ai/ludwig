"""Base loader for HuggingFace translation datasets.

HF translation datasets store source and target sentences in a single
``translation`` column as a dict: ``{'en': '...', 'de': '...'}``.
This loader splits that into two text columns named after the language codes.
"""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


class TranslationLoader(HFLoader):
    """Flatten HF translation dict → two separate text columns.

    Subclasses set ``source_lang`` and ``target_lang`` as class attributes.
    """

    source_lang: str = "en"
    target_lang: str = "de"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        src, tgt = self.source_lang, self.target_lang

        def _extract(row, lang):
            val = row.get("translation", row) if isinstance(row, dict) else {}
            if isinstance(val, dict):
                return str(val.get(lang, ""))
            return str(val)

        if "translation" in df.columns:
            df[src] = df["translation"].apply(lambda x: _extract(x, src))
            df[tgt] = df["translation"].apply(lambda x: _extract(x, tgt))
            df = df.drop(columns=["translation"])

        keep = [src, tgt] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


# ── Per-pair subclasses ───────────────────────────────────────────────────────


class Opus100EnFrLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "fr"


class Opus100EnEsLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "es"


class Wmt14DeEnLoader(TranslationLoader):
    source_lang = "de"
    target_lang = "en"


class Wmt16DeEnLoader(TranslationLoader):
    source_lang = "de"
    target_lang = "en"


class Wmt19DeEnLoader(TranslationLoader):
    source_lang = "de"
    target_lang = "en"


class OpusBooksEnFrLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "fr"


class MultiUNArEnLoader(TranslationLoader):
    source_lang = "ar"
    target_lang = "en"


class EuroparlBgCsLoader(TranslationLoader):
    source_lang = "bg"
    target_lang = "cs"


class EuroparlBgEnLoader(TranslationLoader):
    source_lang = "bg"
    target_lang = "en"


class EuroparlCsEnLoader(TranslationLoader):
    source_lang = "cs"
    target_lang = "en"


class EuroparlDaEnLoader(TranslationLoader):
    source_lang = "da"
    target_lang = "en"


class EuroparlDeEnLoader(TranslationLoader):
    source_lang = "de"
    target_lang = "en"


class EuroparlElEnLoader(TranslationLoader):
    source_lang = "el"
    target_lang = "en"


class EuroparlEnEsLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "es"


class EuroparlEnFrLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "fr"


class EuroparlEnItLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "it"


class EuroparlEnNlLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "nl"


class EuroparlEnPlLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "pl"


class EuroparlEnPtLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "pt"


class EuroparlEnRoLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "ro"


class EuroparlEnSvLoader(TranslationLoader):
    source_lang = "en"
    target_lang = "sv"


class SetimesBgBsLoader(TranslationLoader):
    source_lang = "bg"
    target_lang = "bs"
