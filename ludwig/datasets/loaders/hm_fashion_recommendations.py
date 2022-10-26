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
from typing import List

import numpy as np
import pandas as pd

from ludwig.backend.base import LocalBackend
from ludwig.constants import SPLIT
from ludwig.data.split import get_splitter
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.datasets.loaders.utils import negative_sample


def _merge_dataframes(transactions_df, articles_df, customers_df):
    """Merge the transactions, articles, and customers dataframes into a single dataframe."""
    # Merge the transactions and articles dataframes
    transactions_df = pd.merge(
        transactions_df,
        articles_df,
        how="left",
        left_on="article_id",
        right_on="article_id",
    )

    # Merge the transactions and customers dataframes
    transactions_df = pd.merge(
        transactions_df,
        customers_df,
        how="left",
        left_on="customer_id",
        right_on="customer_id",
    )

    return transactions_df


def _split(df):
    """Split the dataframe into train, validation, and test dataframes.

    The split is done in a chronological manner based on the year_month column. The split is done by customer_id,
    so that interactions for a given customer are present in all splits.

    Params:
        df: The dataframe to split.

    Returns:
        A tuple of (train_df, validation_df, test_df).
    """
    splitter = get_splitter("datetime", column="year_month", probabilities=(0.7, 0.2, 0.1))

    if not isinstance(df, pd.DataFrame):
        df = df.compute()

    train_dfs, val_dfs, test_dfs = [], [], []
    for customer_id in df["customer_id"].unique():
        # Split per customer_id to ensure that interactions for a customer are across all splits
        train_df, val_df, test_df = splitter.split(df[df["customer_id"] == customer_id], backend=LocalBackend())

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


class HMLoader(DatasetLoader):
    def load_unprocessed_dataframe(self, file_paths: List[str], sample=True) -> pd.DataFrame:
        # Load transactions
        df = pd.read_csv(file_paths[2])
        df["t_dat"] = pd.to_datetime(df.t_dat)
        df["year_month"] = df.t_dat.dt.to_period("M").dt.strftime("%Y-%m")

        if sample:
            df = df[df.t_dat > "2020-08-21"]
            customer_ids = np.random.choice(df.customer_id, 100, replace=False)
            df = df[df.customer_id.isin(customer_ids)]

        # 1. Set label to 1 for all known transactions, since the customer bought the article
        df["label"] = 1

        # 2. Split the data into train, validation and test sets. We split per customer_id to ensure that interactions
        # for a customer are across all splits
        train_df, val_df, test_df = _split(df)

        # 3. Negative sample each split separately
        train_df = negative_sample(train_df, neg_pos_ratio=10, neg_val=0, log_pct=100)
        val_df = negative_sample(val_df, neg_pos_ratio=10, neg_val=0, log_pct=100)
        test_df = negative_sample(test_df, neg_pos_ratio=10, neg_val=0, log_pct=100)

        train_df[SPLIT] = 0
        val_df[SPLIT] = 1
        test_df[SPLIT] = 2
        df = pd.concat([train_df, val_df, test_df])

        # 4. Add customer and article features
        articles_df = pd.read_csv(file_paths[0])
        customers_df = pd.read_csv(file_paths[1])
        df = _merge_dataframes(df, articles_df, customers_df)

        # TODO(joppe): add image url once all images are available in a public bucket
        # # Add image url
        # def img_url_or_none(article_id):
        #     url = f"https://h-and-m-kaggle-images.s3.us-west-2.amazonaws.com/{article_id}.jpg"
        #     try:
        #         status_code = requests.head(url, headers={"Access-Control-Request-Method": "GET"}).status_code
        #         return url if status_code == 200 else None
        #     except:
        #         return None
        # df["img_url"] = df["article_id"].apply(img_url_or_none, meta=("img_url", "object"))

        return df
