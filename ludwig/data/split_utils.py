"""Dataset splitting utilities.

Provides functions for splitting datasets into train/validation/test sets using various strategies: random, stratified,
fixed column, datetime, hash.

Extracted from preprocessing.py for modularity. The main split logic remains in ludwig/data/split.py; this module
provides additional utilities.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_split_indices(
    n_samples: int,
    probabilities: tuple[float, float, float] = (0.7, 0.1, 0.2),
    random_seed: int = 42,
) -> np.ndarray:
    """Generate split indices (0=train, 1=validation, 2=test) for a dataset.

    Args:
        n_samples: Number of samples in the dataset.
        probabilities: (train, val, test) split ratios. Must sum to 1.
        random_seed: Random seed for reproducibility.

    Returns:
        Array of split indices (0, 1, or 2) for each sample.
    """
    assert abs(sum(probabilities) - 1.0) < 1e-6, f"Split probabilities must sum to 1, got {sum(probabilities)}"

    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(n_samples)
    splits = np.zeros(n_samples, dtype=int)

    train_end = int(n_samples * probabilities[0])
    val_end = train_end + int(n_samples * probabilities[1])

    splits[indices[train_end:val_end]] = 1
    splits[indices[val_end:]] = 2

    return splits


def stratified_split_indices(
    labels: np.ndarray,
    probabilities: tuple[float, float, float] = (0.7, 0.1, 0.2),
    random_seed: int = 42,
) -> np.ndarray:
    """Generate stratified split indices that maintain label distribution.

    Args:
        labels: Array of class labels for each sample.
        probabilities: (train, val, test) split ratios.
        random_seed: Random seed for reproducibility.

    Returns:
        Array of split indices (0, 1, or 2) for each sample.
    """
    rng = np.random.RandomState(random_seed)
    splits = np.zeros(len(labels), dtype=int)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        n = len(label_indices)
        train_end = int(n * probabilities[0])
        val_end = train_end + int(n * probabilities[1])

        splits[label_indices[train_end:val_end]] = 1
        splits[label_indices[val_end:]] = 2

    return splits
