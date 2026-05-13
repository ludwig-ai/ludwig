"""Loaders for extractive and open-domain QA datasets.

Most HF QA datasets store answers as ``{'text': [...], 'answer_start': [...]}``.
These loaders flatten that to a single string (first answer text, or empty string
for unanswerable questions).
"""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


def _extract_first_answer(answers) -> str:
    """Return the first answer text from various answer formats."""
    if isinstance(answers, dict):
        texts = answers.get("text", answers.get("answer", []))
        if texts:
            first = texts[0] if isinstance(texts, list) else texts
            return str(first)
        return ""
    if hasattr(answers, "__iter__") and not isinstance(answers, str):
        items = list(answers)
        return str(items[0]) if items else ""
    return str(answers)


class SquadLoader(HFLoader):
    """SQuAD v1/v2 extractive QA — context + question → answer text."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["answer"] = df["answers"].apply(_extract_first_answer)
        # For SQuAD v2 unanswerable questions, answer is ""
        keep = ["context", "question", "answer"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class SquadV2Loader(SquadLoader):
    pass


class DuoRCLoader(HFLoader):
    """DuoRC SelfRC — plot + question → answer (first from list)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["answer"] = df["answers"].apply(_extract_first_answer)
        keep = ["plot", "question", "answer"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class HotpotQALoader(HFLoader):
    """HotpotQA — question → answer (ignore supporting facts for simplicity)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        keep = ["question", "answer"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class TriviaQALoader(HFLoader):
    """TriviaQA — question → first answer alias."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["answer_text"] = df["answer"].apply(
            lambda a: str(a.get("value", "")) if isinstance(a, dict) else _extract_first_answer(a)
        )
        keep = ["question", "answer_text"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class PubMedQALoader(HFLoader):
    """PubMedQA — context (joined) + question → yes/no/maybe label."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # context is a dict with 'contexts' list
        df["context_text"] = df["context"].apply(
            lambda c: " ".join(c.get("contexts", [])) if isinstance(c, dict) else str(c)
        )
        keep = ["context_text", "question", "final_decision"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class FeverGoldLoader(HFLoader):
    """FEVER gold evidence — claim → verdict (SUPPORTS/REFUTES/NEI)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # evidence is a list of lists — just keep the claim and label
        keep = ["claim", "label"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class NaturalQuestionsLoader(HFLoader):
    """Natural Questions simplified — question + short answer."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # NQ has very complex nested structure; use just question_text and annotations
        if "question" in df.columns and isinstance(df["question"].iloc[0], dict):
            df["question_text"] = df["question"].apply(lambda q: q.get("text", "") if isinstance(q, dict) else str(q))
        elif "question_text" in df.columns:
            pass
        else:
            df["question_text"] = df.get("question", df.iloc[:, 0]).astype(str)

        if "annotations" in df.columns:
            df["answer_text"] = df["annotations"].apply(
                lambda a: (a.get("short_answers", [{}])[0] or {}).get("text", [""])[0] if isinstance(a, dict) else ""
            )
        else:
            df["answer_text"] = ""

        keep = ["question_text", "answer_text"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
