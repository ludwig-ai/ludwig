"""California Housing dataset loader.

Wraps sklearn.datasets.fetch_california_housing to provide the standard ML regression benchmark through Ludwig's dataset
infrastructure.
"""

import numpy as np
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class CaliforniaHousingLoader(DatasetLoader):
    def load(self, kaggle_username=None, kaggle_key=None, split=False):
        """Load California Housing directly from sklearn."""
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        # Add deterministic split column (70/15/15)
        n = len(df)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        splits = np.zeros(n, dtype=int)
        splits[indices[int(n * 0.7) : int(n * 0.85)]] = 1
        splits[indices[int(n * 0.85) :]] = 2
        df["split"] = splits

        if split:
            return self.split(df)
        return df
