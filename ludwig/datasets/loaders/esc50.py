import os

import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class ESC50Loader(DatasetLoader):
    """ESC-50 Environmental Sound Classification dataset.

    After extraction the archive contains:
        ESC-50-master/audio/*.wav   — 2,000 WAV clips
        ESC-50-master/meta/esc50.csv — metadata (filename, fold, target, category, …)

    The loader rewrites the ``filename`` column into an ``audio_path`` column with
    paths relative to the processed dataset directory (where ``preserve_paths``
    copies the audio subdirectory).
    """

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_dataframe(dataframe)
        # Build relative paths so the smoke-test can prepend processed_dataset_dir.
        df["audio_path"] = df["filename"].apply(
            lambda fn: os.path.join("ESC-50-master", "audio", os.path.basename(str(fn)))
        )
        # Use integer fold (1-5) as the split column: fold 5 → test, fold 4 → val, rest → train.
        df["split"] = df["fold"].apply(lambda f: 2 if f == 5 else (1 if f == 4 else 0))
        return df
