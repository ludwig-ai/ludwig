"""Base loader for multi-label classification datasets where labels are stored
as a list of integers or strings. Ludwig's set feature expects a
space-separated string of label names.
"""

from __future__ import annotations

import ast

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


def _coerce_to_list(val) -> list:
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val.split()
    if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
        return list(val)
    return [val]


class MultiLabelLoader(HFLoader):
    """Convert a list-of-labels column to a space-separated string.

    Subclasses set:
        labels_col   – column containing the label list
        label_names  – list mapping int index → label string (or None to use as-is)
        out_col      – output column name
    """

    labels_col: str = "labels"
    label_names: list[str] | None = None
    out_col: str = "labels"

    def _map_label(self, idx) -> str:
        if self.label_names is not None:
            try:
                return self.label_names[int(idx)]
            except (IndexError, ValueError):
                return str(idx)
        return str(idx)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.out_col] = df[self.labels_col].apply(lambda v: " ".join(self._map_label(x) for x in _coerce_to_list(v)))
        keep = [c for c in df.columns if c not in (self.labels_col,) or c == self.out_col]
        return df[keep]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


# LexGLUE ECtHR — 10 article labels (int indices)
_ECTHR_LABELS = [
    "Art. 2",
    "Art. 3",
    "Art. 5",
    "Art. 6",
    "Art. 8",
    "Art. 9",
    "Art. 10",
    "Art. 11",
    "Art. 14",
    "Art. P1-1",
]


class LexGlueECtHRLoader(HFLoader):
    """LexGLUE ECtHR — text (list of paragraphs joined) → label set."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # text is a list of paragraphs
        df["text_joined"] = df["text"].apply(
            lambda v: (
                " ".join(str(p) for p in _coerce_to_list(v))
                if hasattr(v, "__iter__") and not isinstance(v, str)
                else str(v)
            )
        )
        df["labels"] = df["labels"].apply(
            lambda v: " ".join(
                _ECTHR_LABELS[int(i)] if int(i) < len(_ECTHR_LABELS) else str(i) for i in _coerce_to_list(v)
            )
        )
        keep = ["text_joined", "labels"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class LexGlueEURLexLoader(HFLoader):
    """LexGLUE EuroVoc — text → space-separated EuroVoc concept IDs."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda v: " ".join(str(x) for x in _coerce_to_list(v)))
        keep = ["text", "labels"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class BeaverTailsLoader(HFLoader):
    """BeaverTails safety — response + prompt → is_safe binary, category string."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "category" in df.columns and df["category"].dtype == object:
            # category is a dict of {label: bool} — find active labels
            df["category_labels"] = df["category"].apply(
                lambda c: " ".join(k for k, v in c.items() if v) if isinstance(c, dict) else str(c)
            )
        keep = (
            ["prompt", "response", "is_safe", "category_labels"]
            if "category_labels" in df.columns
            else ["prompt", "response", "is_safe"]
        )
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
