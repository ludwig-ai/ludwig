"""Miscellaneous loaders for datasets that don't fit other base patterns."""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


class KlueStsLoader(HFLoader):
    """KLUE STS — sentence pair → similarity score (regression)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # labels is a dict {'label': float, 'real-label': float}
        df["score"] = df["labels"].apply(lambda v: float(v.get("label", 0.0)) if isinstance(v, dict) else float(v))
        keep = ["sentence1", "sentence2", "score"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class MultiRCLoader(HFLoader):
    """SuperGLUE MultiRC — paragraph + question + answer → label (0/1)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # idx is a nested dict; we don't need it
        keep = ["paragraph", "question", "answer", "label"]
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class TruthfulQALoader(HFLoader):
    """TruthfulQA multiple choice — question → best_answer category."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # mc1_targets / mc2_targets are dicts; just use best_answer as text
        keep = ["question", "best_answer", "category"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class GiftEvalLoader(HFLoader):
    """GiftEval Pretrain — time-series forecasting: use freq + item_id as features."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # target is a list of floats; use the mean as a scalar regression target
        if "target" in df.columns:
            df["target_mean"] = df["target"].apply(
                lambda v: float(sum(v) / len(v)) if hasattr(v, "__iter__") and len(v) > 0 else 0.0
            )
        keep = ["freq", "item_id", "target_mean"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class HC3Loader(HFLoader):
    """HC3 — expand each (question, human_answers, chatgpt_answers) row into two rows
    for binary classification: detect if an answer is human-written (0) or ChatGPT (1).
    Takes the first element from each answer list.
    """

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def first(lst):
            if isinstance(lst, list) and lst:
                return str(lst[0])
            return ""

        human = pd.DataFrame(
            {
                "question": df["question"],
                "answer": df["human_answers"].apply(first),
                "is_chatgpt": 0,
            }
        )
        chatgpt = pd.DataFrame(
            {
                "question": df["question"],
                "answer": df["chatgpt_answers"].apply(first),
                "is_chatgpt": 1,
            }
        )
        return pd.concat([human, chatgpt], ignore_index=True)


class BlimpLoader(HFLoader):
    """BLiMP — reshape minimal pairs into binary grammaticality classification.

    Each source row has (sentence_good, sentence_bad). We expand to two rows
    each with a single `sentence` column and a binary `is_grammatical` label.
    """

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        good = df[["sentence_good"]].rename(columns={"sentence_good": "sentence"})
        good["is_grammatical"] = 1
        bad = df[["sentence_bad"]].rename(columns={"sentence_bad": "sentence"})
        bad["is_grammatical"] = 0
        return pd.concat([good, bad], ignore_index=True)


class SciqLoader(HFLoader):
    """SciQ — support + question + distractor/correct_answer → correct answer (text)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        keep = ["support", "question", "correct_answer"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
