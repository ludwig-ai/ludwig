import logging
import time
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import scipy

from ludwig.utils.types import DataFrame


def _negative_sample_user(interaction_row: np.array, neg_pos_ratio: int, extra_samples: int) -> Tuple[List[int], int]:
    """Returns a list of negative item indices for given user-item interactions.

    If there are not enough negative items, takes all of them and adds the difference to the extra_samples
    otherwise, samples with replacement.

    Params:
        interaction_row: user-item interaction row
        neg_pos_ratio: number of negative samples per positive sample
        extra_samples: number of additional samples to add to the negative sample list
    Returns:
        Tuple of list of negative item indices and number of extra samples
    """
    # Find all items that are not interacted with by the user
    neg_items = np.where(interaction_row == 0)[1]
    available_samples = len(neg_items)

    # Randomly sample negative items
    npos = interaction_row.shape[1] - len(neg_items)
    samples_required = npos * neg_pos_ratio + extra_samples
    should_sample = samples_required <= available_samples

    neg_items = np.random.choice(neg_items, samples_required, replace=False) if should_sample else neg_items

    return neg_items.tolist(), max(0, samples_required - available_samples)


def negative_sample(
    df: DataFrame,
    user_id_col: str = "customer_id",
    item_id_col: str = "article_id",
    label_col: str = "label",
    neg_pos_ratio: int = 1,
    neg_val: Any = 0,
    log_pct: int = 0,
):
    """Negative sampling for implicit feedback datasets.

    Params:
        df: DataFrame containing user-item interactions
        user_id_col: column name for user ids
        item_id_col: column name for item ids
        label_col: column name for interaction labels (e.g. 1 for positive interaction)
        n_neg: number of negative samples per positive sample
        neg_val: label value for the negative samples
        percent_print: print progress every percent_print percent. 0 to disable
    Returns:
        Input DataFrame with negative samples appended

    Source: https://petamind.com/fast-uniform-negative-sampling-for-rating-matrix/
    """
    # TODO(joppe): support out of memory negative sampling using Dask
    if not isinstance(df, pd.DataFrame):
        df = df.compute()

    # Initialize sparse COOrdinate matrix from users and items in existing interactions
    user_id_cat = df[user_id_col].astype("category").cat
    user_id_codes = user_id_cat.codes.values

    item_id_cat = df[item_id_col].astype("category").cat
    item_id_codes = item_id_cat.codes.values

    interactions_sparse = scipy.sparse.coo_matrix((df[label_col], (user_id_codes, item_id_codes)))

    # Convert to dense user-item matrix so we can iterate
    interactions_dense = interactions_sparse.todense()

    nrows = interactions_dense.shape[0]
    niter_log = int(nrows * log_pct / 100)
    start_time = time.time()

    user_indices, item_indices = [], []
    extra_samples = 0
    for user_idx, interaction_row in enumerate(interactions_dense):
        if log_pct > 0 and user_idx % niter_log == 0:
            logging.info(
                f"Negative sampling progress: {float(user_idx) * 100 / nrows:0.0f}% in {time.time() - start_time:0.2f}s"
            )

        neg_items_for_user, extra_samples = _negative_sample_user(interaction_row, neg_pos_ratio, extra_samples)

        # Add to negative user-item pairs
        item_indices += neg_items_for_user
        user_indices += [user_idx] * len(neg_items_for_user)

    negative_samples = pd.DataFrame(
        {
            # Map back to original user and item ids
            user_id_col: user_id_cat.categories[user_indices],
            item_id_col: item_id_cat.categories[item_indices],
            label_col: [neg_val] * len(item_indices),
        }
    )

    return pd.concat([df[[user_id_col, item_id_col, label_col]], negative_samples])
