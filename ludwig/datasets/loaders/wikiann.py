import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader

# WikiANN IOB2 tag vocabulary (integer index → string label).
_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class WikiANNLoader(HFLoader):
    """WikiANN English NER dataset.

    The raw HuggingFace dataset stores tokens and ner_tags as Python lists.
    Ludwig's text/sequence features expect space-separated strings, so this
    loader joins the lists and maps integer NER tag ids to IOB2 label strings.
    """

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sentence"] = df["tokens"].apply(
            lambda toks: (
                " ".join(str(t) for t in toks) if hasattr(toks, "__iter__") and not isinstance(toks, str) else str(toks)
            )
        )
        df["ner_tags"] = df["ner_tags"].apply(
            lambda tags: (
                " ".join(_NER_LABELS[int(t)] if int(t) < len(_NER_LABELS) else "O" for t in tags)
                if hasattr(tags, "__iter__") and not isinstance(tags, str)
                else str(tags)
            )
        )
        keep = ["sentence", "ner_tags"] + (["split"] if "split" in df.columns else [])
        return df[keep]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return self._transform(train), self._transform(val), self._transform(test)
        df = super().load(split=False)
        return self._transform(df)
