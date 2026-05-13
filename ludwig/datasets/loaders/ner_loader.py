"""Base loader for token-level NER / sequence-labeling datasets.

HuggingFace NER datasets store tokens and tags as lists (or numpy arrays).
This loader joins them into space-separated strings that Ludwig's
text/sequence features expect.
"""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


def _to_str_list(val) -> list[str]:
    """Coerce any iterable (list, tuple, numpy array) to a list of strings."""
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return [str(v) for v in val]
    return str(val).split()


def _to_int_list(val) -> list[int]:
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return [int(v) for v in val]
    s = str(val).strip().strip("[]")
    return [int(t) for t in s.split() if t]


class NERLoader(HFLoader):
    """Convert token/tag lists to space-separated strings.

    Subclasses set:
        tokens_col  – column with the token list
        tags_col    – column with integer tag indices
        tag_labels  – list mapping index → string label
        out_sentence_col – output column name for the sentence
        out_tags_col     – output column name for the tag sequence
    """

    tokens_col: str = "tokens"
    tags_col: str = "ner_tags"
    tag_labels: list[str] = []
    out_sentence_col: str = "sentence"
    out_tags_col: str = "ner_tags"

    def _map_tag(self, t: int) -> str:
        if self.tag_labels and t < len(self.tag_labels):
            return self.tag_labels[t]
        return str(t)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.out_sentence_col] = df[self.tokens_col].apply(lambda v: " ".join(_to_str_list(v)))
        df[self.out_tags_col] = df[self.tags_col].apply(lambda v: " ".join(self._map_tag(t) for t in _to_int_list(v)))
        keep = [self.out_sentence_col, self.out_tags_col]
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


# ── Per-dataset subclasses ────────────────────────────────────────────────────

_WIKIANN_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class WikiannEnLoader(NERLoader):
    tags_col = "ner_tags"
    tag_labels = _WIKIANN_TAGS


class WikiannDeLoader(NERLoader):
    tags_col = "ner_tags"
    tag_labels = _WIKIANN_TAGS


class WikiannZhLoader(NERLoader):
    tags_col = "ner_tags"
    tag_labels = _WIKIANN_TAGS


# MultiNERD: 31-class NER (PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD,
# INST, MEDIA, MYTH, PLANT, TIME, VEHI + B-/I- prefixes + O)
_MULTINERD_TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]


class MultiNERDLoader(NERLoader):
    tag_labels = _MULTINERD_TAGS


# FewNERD: fine-grained NER — tags are already string labels in this dataset,
# but can also be integer indices. We'll keep them as-is strings.
class FewNERDLoader(HFLoader):
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sentence"] = df["tokens"].apply(lambda v: " ".join(_to_str_list(v)))
        df["ner_tags"] = df["ner_tags"].apply(lambda v: " ".join(str(t) for t in _to_int_list(v)))
        keep = ["sentence", "ner_tags"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class AcronymIdentificationLoader(NERLoader):
    tags_col = "labels"
    tag_labels = ["O", "B-long", "I-long", "B-short", "I-short"]
    out_tags_col = "labels"


class PIIMaskingLoader(HFLoader):
    """PII Masking — keep source_text and mbert_bio_labels (space-joined)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bio_labels"] = df["mbert_bio_labels"].apply(lambda v: " ".join(_to_str_list(v)))
        keep = ["source_text", "bio_labels"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class WinobiasLoader(NERLoader):
    """WinoBias coref — keep just the sentence (joined tokens) + gender label."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sentence"] = df["tokens"].apply(lambda v: " ".join(_to_str_list(v)))
        keep = ["sentence", "label"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]
