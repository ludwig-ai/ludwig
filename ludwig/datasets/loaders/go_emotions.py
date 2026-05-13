import ast

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader

# 28-class emotion vocabulary (index → label string).
_EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


class GoEmotionsLoader(HFLoader):
    """GoEmotions multi-label emotion classification dataset.

    The HuggingFace 'simplified' split stores ``labels`` as a list of integer
    indices.  Ludwig's set feature expects a space-separated string of category
    names, so this loader maps the integer ids to emotion label strings.
    """

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _map_ids(ids):
            if isinstance(ids, str):
                try:
                    ids = ast.literal_eval(ids)
                except (ValueError, SyntaxError):
                    return ids
            # Covers list, tuple, and numpy arrays
            try:
                return " ".join(_EMOTION_LABELS[int(i)] if int(i) < len(_EMOTION_LABELS) else str(i) for i in ids)
            except TypeError:
                return str(ids)

        df["labels"] = df["labels"].apply(_map_ids)
        keep = ["text", "labels"] + (["split"] if "split" in df.columns else [])
        return df[keep]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        df = super().load(split=False)
        return self._transform(df)
