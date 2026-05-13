import os

import pandas as pd

from ludwig.datasets.loaders.hugging_face import HFLoader

_SPLITS = {"train": 0, "validation": 1, "test": 2}


class NewYorkerCaptionContestLoader(HFLoader):
    """New Yorker Caption Contest (matching config) — multimodal image+text classification.

    The HuggingFace dataset stores images as PIL objects.  This loader saves them
    as JPEG files under ``<processed_dataset_dir>/images/`` (with caching) and
    returns a DataFrame with absolute ``image_path`` and ``image_description``
    columns alongside a ``label`` target column.
    """

    def _save_images(self, df: pd.DataFrame, split_name: str) -> pd.Series:
        """Save PIL images to disk and return a Series of absolute paths."""
        img_dir = os.path.join(self.processed_dataset_dir, "images", split_name)
        os.makedirs(img_dir, exist_ok=True)

        paths = []
        for idx, row in df.iterrows():
            img_path = os.path.join(img_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                img = row["image"]
                if hasattr(img, "save"):
                    img.save(img_path, format="JPEG")
                elif isinstance(img, dict) and "bytes" in img:
                    import io

                    from PIL import Image

                    Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(img_path, format="JPEG")
            paths.append(img_path)
        return pd.Series(paths, index=df.index)

    def _transform(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["image_path"] = self._save_images(df, split_name)
        keep = ["image_path", "image_description", "label"]
        if "split" in df.columns:
            keep.append("split")
        return df[keep]

    def load(self, split: bool = False):
        if split:
            train, val, test = super().load(split=True)
            return (
                self._transform(train, "train"),
                self._transform(val, "validation"),
                self._transform(test, "test"),
            )
        df = super().load(split=False)
        # super() adds integer split column; reconstruct per-split name for image dirs
        result_parts = []
        for split_int, split_name in [(0, "train"), (1, "validation"), (2, "test")]:
            part = df[df["split"] == split_int]
            if not part.empty:
                result_parts.append(self._transform(part, split_name))
        return pd.concat(result_parts).reset_index(drop=True)
