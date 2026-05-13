"""Loaders for code/programming datasets where token lists need joining."""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


def _join_tokens(val) -> str:
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return " ".join(str(t) for t in val)
    return str(val)


class CodeSearchNetLoader(HFLoader):
    """CodeSearchNet — func_code_string + func_documentation_string."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Use the pre-joined string columns if available
        keep = []
        for col in ["func_code_string", "func_documentation_string", "language"]:
            if col in df.columns:
                keep.append(col)
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class CodeXGlueLoader(HFLoader):
    """CodeXGlue code-to-text — code → docstring."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "code_tokens" in df.columns:
            df["code"] = df["code_tokens"].apply(_join_tokens)
        if "docstring_tokens" in df.columns:
            df["docstring"] = df["docstring_tokens"].apply(_join_tokens)
        keep = ["code", "docstring"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class MBPPLoader(HFLoader):
    """Mostly Basic Python Problems — prompt → code (text generation)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        keep = ["text", "code"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
