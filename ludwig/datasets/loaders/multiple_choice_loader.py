"""Base loader for multiple-choice QA datasets.

Multiple-choice datasets store answer options as a list or dict.
This loader linearizes them into a single ``choices_text`` column
(format: "A: ... B: ... C: ... D: ...") alongside the question,
and maps the correct answer to a letter label (A/B/C/D/E).
"""

from __future__ import annotations

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader

_LABELS = list("ABCDEFGHIJ")


def _linearize_choices(choices) -> str:
    """Convert a list or dict of choices to 'A: x B: y ...' string."""
    if isinstance(choices, dict):
        # Format: {'text': [...], 'label': [...]} (ARC, OpenBookQA)
        texts = choices.get("text", choices.get("choices", []))
        labels = choices.get("label", _LABELS[: len(texts)])
        return " ".join(f"{lbl}: {t}" for lbl, t in zip(labels, texts))
    if hasattr(choices, "__iter__") and not isinstance(choices, str):
        return " ".join(f"{_LABELS[i]}: {t}" for i, t in enumerate(choices))
    return str(choices)


def _answer_to_int(answer, choices) -> int:
    """Convert an answer label/index to 0-based int."""
    if isinstance(answer, int):
        return answer
    s = str(answer).strip()
    if s in _LABELS:
        return _LABELS.index(s)
    # try numeric string
    try:
        return int(s)
    except ValueError:
        pass
    # try matching against choices text
    if hasattr(choices, "__iter__") and not isinstance(choices, (str, dict)):
        for i, c in enumerate(choices):
            if str(c).strip() == s:
                return i
    return 0


class MultipleChoiceLoader(HFLoader):
    """Flatten multiple-choice QA datasets into question + choices_text + label columns.

    Subclasses must set:
        question_col  – column with the question text
        choices_col   – column with the answer choices (list or dict)
        answer_col    – column with the correct answer (letter or index)
        context_col   – optional additional context column (or None)
    """

    question_col: str = "question"
    choices_col: str = "choices"
    answer_col: str = "answer"
    context_col: str | None = None

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["choices_text"] = df[self.choices_col].apply(_linearize_choices)

        # Map answer to letter (A/B/C/D)
        def _to_letter(row):
            idx = _answer_to_int(row[self.answer_col], row.get(self.choices_col))
            return _LABELS[idx] if idx < len(_LABELS) else "A"

        df["answer_label"] = df.apply(_to_letter, axis=1)

        keep = []
        if self.context_col and self.context_col in df.columns:
            keep.append(self.context_col)
        keep += [self.question_col, "choices_text", "answer_label"]
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


# ── Per-dataset subclasses ────────────────────────────────────────────────────


class HellaSwagLoader(MultipleChoiceLoader):
    question_col = "ctx"
    choices_col = "endings"
    answer_col = "label"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["choices_text"] = df["endings"].apply(_linearize_choices)

        def _to_letter(label):
            try:
                return _LABELS[int(label)]
            except (ValueError, IndexError):
                return "A"

        df["answer_label"] = df["label"].apply(_to_letter)
        keep = ["activity_label", "ctx", "choices_text", "answer_label"]
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]


class CommonsenseQALoader(MultipleChoiceLoader):
    question_col = "question"
    choices_col = "choices"
    answer_col = "answerKey"


class ArcLoader(MultipleChoiceLoader):
    question_col = "question"
    choices_col = "choices"
    answer_col = "answerKey"


class OpenBookQALoader(MultipleChoiceLoader):
    question_col = "question_stem"
    choices_col = "choices"
    answer_col = "answerKey"
    context_col = "fact1"


class MmluLoader(MultipleChoiceLoader):
    question_col = "question"
    choices_col = "choices"
    answer_col = "answer"


class MmluProLoader(MultipleChoiceLoader):
    question_col = "question"
    choices_col = "options"
    answer_col = "answer"
    context_col = "cot_content"


class ScienceQALoader(MultipleChoiceLoader):
    question_col = "question"
    choices_col = "choices"
    answer_col = "answer"
    context_col = "lecture"


class BbhLoader(HFLoader):
    """Big-Bench Hard — input/target are plain strings, no choices."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        keep = ["input", "target"] + (["split"] if "split" in df.columns else [])
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
