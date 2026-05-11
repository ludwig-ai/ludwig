# Copyright (c) 2022 Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ludwig.constants import SPLIT
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class OpenMLLoader(DatasetLoader):
    """Loads any OpenML task by task ID.

    Unlike other DatasetLoaders, OpenMLLoader bypasses the usual download/extract/transform pipeline and fetches
    data directly via the OpenML Python API. The processed dataset is cached as a Parquet file so that subsequent
    calls skip the network round-trip.

    The task ID can be provided either in the DatasetConfig (``config.openml_task_id``) or passed directly to
    :meth:`load` via the ``openml_task_id`` keyword argument (which takes precedence).
    """

    def load(
        self, openml_task_id: int | None = None, split: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load an OpenML task, downloading and caching it if necessary.

        :param openml_task_id: (int, optional) OpenML task ID. Overrides ``self.config.openml_task_id`` when given.
        :param split: (bool) If True, return a 3-tuple of (train_df, val_df, test_df). If False (default), return a
            single DataFrame with a ``split`` column (0=train, 1=validation, 2=test).
        :returns: A single DataFrame or a 3-tuple of DataFrames depending on the value of ``split``.
        """
        task_id = openml_task_id if openml_task_id is not None else self.config.openml_task_id
        if task_id is None:
            raise ValueError("No OpenML task ID provided. Set config.openml_task_id or pass openml_task_id= to load().")

        if os.path.exists(self.processed_dataset_path):
            # Validate that the cached file was written for this task ID.
            import pyarrow.parquet as pq

            cached_meta = pq.read_metadata(self.processed_dataset_path).metadata or {}
            cached_task_id = cached_meta.get(b"openml_task_id", b"").decode()
            if cached_task_id and int(cached_task_id) != int(task_id):
                logger.warning(f"Cached file was for task {cached_task_id}, but requested task {task_id}. Re-fetching.")
                os.remove(self.processed_dataset_path)
                df = self._fetch_and_cache(task_id)
            else:
                logger.info(f"Loading cached OpenML task {task_id} from {self.processed_dataset_path}")
                df = pd.read_parquet(self.processed_dataset_path)
        else:
            df = self._fetch_and_cache(task_id)

        if split:
            df[SPLIT] = pd.to_numeric(df[SPLIT])
            train_df = df[df[SPLIT] == 0].drop(columns=[SPLIT]).reset_index(drop=True)
            val_df = df[df[SPLIT] == 1].drop(columns=[SPLIT]).reset_index(drop=True)
            test_df = df[df[SPLIT] == 2].drop(columns=[SPLIT]).reset_index(drop=True)
            return train_df, val_df, test_df
        return df

    def _fetch_and_cache(self, task_id: int) -> pd.DataFrame:
        """Download the OpenML task, build a DataFrame with a split column, and save it to Parquet.

        :param task_id: (int) OpenML task ID.
        :returns: The combined DataFrame (with ``split`` column).
        """
        try:
            import openml
        except ImportError:
            raise ImportError("openml package is required: pip install openml")

        logger.info(f"Fetching OpenML task {task_id} …")
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)

        # Newer openml versions already return X as a pd.DataFrame; older versions return a numpy array.
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=attribute_names if attribute_names else list(range(X.shape[1])))

        # Ensure the target column is present under its original name.
        target_col = task.target_name if task.target_name else "target"
        if isinstance(y, pd.Series):
            df[target_col] = y.values
        else:
            df[target_col] = y

        df = self._assign_splits(df, task)

        os.makedirs(self.processed_dataset_dir, exist_ok=True)
        logger.info(f"Saving processed OpenML task {task_id} to {self.processed_dataset_path}")
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing_meta, b"openml_task_id": str(task_id).encode()})
        pq.write_table(table, self.processed_dataset_path)
        return df

    @staticmethod
    def _assign_splits(df: pd.DataFrame, task) -> pd.DataFrame:
        """Assign a ``split`` column (0=train, 1=val, 2=test) to *df*.

        Tries to use the OpenML task's predefined train/test split (fold 0). If the split is unavailable, falls back to
        a random 70/10/20 split.

        :param df: Combined feature + target DataFrame (no split column yet).
        :param task: OpenML task object.
        :returns: The same DataFrame with a ``split`` column added.
        """
        n = len(df)
        split_col = np.full(n, 0, dtype=np.int8)  # Default: all train

        try:
            train_indices, test_indices = task.get_train_test_split_indices(fold=0)
            split_col[test_indices] = 2

            # Carve a validation set out of the training indices (≈12.5 % of train → ~10 % of total).
            rng = np.random.default_rng(seed=42)
            val_size = max(1, int(0.125 * len(train_indices)))
            val_indices = rng.choice(train_indices, size=val_size, replace=False)
            split_col[val_indices] = 1

            logger.info(
                f"Used OpenML predefined split: {(split_col == 0).sum()} train, "
                f"{(split_col == 1).sum()} val, {(split_col == 2).sum()} test."
            )
        except Exception as exc:
            logger.warning(f"Could not retrieve OpenML predefined split ({exc}). Falling back to random 70/10/20.")
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n)
            train_end = int(0.70 * n)
            val_end = int(0.80 * n)
            split_col[indices[train_end:val_end]] = 1
            split_col[indices[val_end:]] = 2

        df = df.copy()
        df[SPLIT] = split_col
        return df


def openml_suite_loaders(suite_id: int, cache_dir: str | None = None) -> list[OpenMLLoader]:
    """Return a list of :class:`OpenMLLoader` instances for all tasks in an OpenML benchmark suite.

    Well-known suite IDs:

    * **99**  – OpenML-CC18 (72 classification tasks)
    * **269** – OpenML regression suite
    * **271** – OpenML-CTR23 (35 regression tasks)
    * **337** – OpenML-AutoML (classification)

    :param suite_id: (int) OpenML benchmark suite (study) ID.
    :param cache_dir: (str, optional) Directory used to cache downloaded datasets. Defaults to the Ludwig cache
        location when *None*.
    :returns: A list of :class:`OpenMLLoader` instances, one per task in the suite.
    """
    try:
        import openml
    except ImportError:
        raise ImportError("openml package is required: pip install openml")

    suite = openml.study.get_suite(suite_id)
    loaders: list[OpenMLLoader] = []
    for task_id in suite.tasks:
        config = DatasetConfig(
            name=f"openml_task_{task_id}",
            version="1",
            description=f"OpenML task {task_id} from suite {suite_id}",
            openml_task_id=task_id,
            loader="openml_loader.OpenMLLoader",
        )
        loaders.append(OpenMLLoader(config=config, cache_dir=cache_dir))
    return loaders
