"""Loaders for Visual Question Answering datasets with embedded images.

The Cauldron datasets store images as a list of PIL objects in an ``images``
column and conversations/questions in a ``texts`` column.  This loader saves
the first image to disk (with caching) and flattens the QA text.
"""

from __future__ import annotations

import io
import os

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader


class CauldronVQALoader(HFLoader):
    """Base loader for HuggingFaceM4/the_cauldron VQA subsets.

    Images (list of PIL) → saved as JPEG, first image path returned.
    Texts (list of dicts with 'user'/'assistant' keys) → question + answer.
    """

    def _save_images(self, df: pd.DataFrame, split_name: str) -> pd.Series:
        img_dir = os.path.join(self.processed_dataset_dir, "images", split_name)
        os.makedirs(img_dir, exist_ok=True)
        paths = []
        for idx, row in df.iterrows():
            img_path = os.path.join(img_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                images = row.get("images", [])
                if hasattr(images, "__iter__") and not isinstance(images, (str, dict)):
                    img_list = list(images)
                    if img_list:
                        img = img_list[0]
                        self._save_pil(img, img_path)
            paths.append(img_path)
        return pd.Series(paths, index=df.index)

    @staticmethod
    def _save_pil(img, path: str) -> None:
        if hasattr(img, "save"):
            img.convert("RGB").save(path, format="JPEG")
        elif isinstance(img, dict) and "bytes" in img:
            from PIL import Image

            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, format="JPEG")

    @staticmethod
    def _extract_qa(texts) -> tuple[str, str]:
        if hasattr(texts, "__iter__") and not isinstance(texts, (str, dict)):
            items = list(texts)
            question, answer = "", ""
            for item in items:
                if isinstance(item, dict):
                    if "user" in item:
                        question = str(item["user"])
                    if "assistant" in item:
                        answer = str(item["assistant"])
            return question, answer
        return str(texts), ""

    def _transform(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["image_path"] = self._save_images(df, split_name)
        qa = df["texts"].apply(self._extract_qa)
        df["question"] = qa.apply(lambda x: x[0])
        df["answer"] = qa.apply(lambda x: x[1])
        keep = ["image_path", "question", "answer"]
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return (
                self._transform(train, "train"),
                self._transform(val, "validation"),
                self._transform(test, "test"),
            )
        df = super().load(split=False)
        parts = []
        for split_int, split_name in [(0, "train"), (1, "validation"), (2, "test")]:
            part = df[df["split"] == split_int]
            if not part.empty:
                parts.append(self._transform(part, split_name))
        return pd.concat(parts).reset_index(drop=True) if parts else self._transform(df, "train")


class AI2DiagramsLoader(CauldronVQALoader):
    pass


class TextVQALoader(CauldronVQALoader):
    pass


class VQAv2Loader(CauldronVQALoader):
    pass


class DocVQALoader(CauldronVQALoader):
    pass


class MmmuLoader(HFLoader):
    """MMMU — image(s) + question → answer (multiple choice)."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # MMMU has image_1..image_7 columns; keep first non-null
        for img_col in [f"image_{i}" for i in range(1, 8)]:
            if img_col in df.columns:
                df["image"] = df[img_col]
                break
        keep = ["question", "options", "answer"]
        if "image" in df.columns:
            keep = ["image"] + keep
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class MathVistaLoader(HFLoader):
    """MathVista — image + question → answer."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "choices" in df.columns:
            from ludwig.datasets.loaders.multiple_choice_loader import _linearize_choices

            df["choices_text"] = df["choices"].apply(_linearize_choices)
        keep = ["image", "question", "answer"]
        if "choices_text" in df.columns:
            keep.insert(2, "choices_text")
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))


class ScienceQAImageLoader(HFLoader):
    """ScienceQA — optional image + question + choices → answer letter."""

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from ludwig.datasets.loaders.multiple_choice_loader import _LABELS, _linearize_choices

        df["choices_text"] = df["choices"].apply(_linearize_choices)

        def _to_letter(ans):
            try:
                return _LABELS[int(ans)]
            except (ValueError, IndexError, TypeError):
                return "A"

        df["answer_label"] = df["answer"].apply(_to_letter)
        keep = ["question", "choices_text", "answer_label"]
        for col in ["hint", "lecture", "solution"]:
            if col in df.columns:
                keep.insert(1, col)
                break
        if "split" in df.columns:
            keep.append("split")
        return df[[c for c in keep if c in df.columns]]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        return self._transform(super().load(split=False))
