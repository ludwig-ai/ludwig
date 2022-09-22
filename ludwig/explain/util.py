from dataclasses import dataclass, field
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, PREPROCESSING, SPLIT


@dataclass
class LabelExplanation:
    """Stores the feature attributions for a single label in the target feature's vocab."""

    # The attribution for each input feature.
    feature_attributions: npt.NDArray[np.float64]


@dataclass
class Explanation:
    """Stores the explanations for a single row of input data.

    Contains the feature attributions for each label in the target feature's vocab.
    """

    target: str

    # The explanations for each label in the vocab of the target feature.
    label_explanations: List[LabelExplanation] = field(default_factory=list)

    def add(self, feature_attributions: npt.NDArray[np.float64]):
        """Add the feature attributions for a single label."""
        if len(self.label_explanations) > 0:
            # Check that the feature attributions are the same shape as existing explanations.
            assert self.label_explanations[0].feature_attributions.shape == feature_attributions.shape, (
                f"Expected feature attributions of shape {self.label_explanations[0].feature_attributions.shape}, "
                f"got {feature_attributions.shape}"
            )
        self.label_explanations.append(LabelExplanation(feature_attributions))

    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert the explanation to a 2D array of shape (num_labels, num_features)."""
        return np.array([le.feature_attributions for le in self.label_explanations])


def filter_cols(df, cols):
    cols = {c.lower() for c in cols}
    retain_cols = [c for c in df.columns if c.lower() in cols]
    return df[retain_cols]


def prepare_data(model: LudwigModel, inputs_df: pd.DataFrame, sample_df: pd.DataFrame, target: str):
    feature_cols = [feature[COLUMN] for feature in model.config[INPUT_FEATURES]]
    if SPLIT in model.config.get(PREPROCESSING, {}) and COLUMN in model.config[PREPROCESSING][SPLIT]:
        feature_cols.append(model.config[PREPROCESSING][SPLIT][COLUMN])
    target_feature_name = get_feature_name(model, target)

    inputs_df = filter_cols(inputs_df, feature_cols)
    if sample_df is not None:
        sample_df = filter_cols(sample_df, feature_cols)

    return inputs_df, sample_df, feature_cols, target_feature_name


def get_pred_col(preds, target):
    t = target.lower()
    for c in preds.keys():
        if c.lower() == t:
            if "probabilities" in preds[c]:
                return preds[c]["probabilities"]
            else:
                return preds[c]["predictions"]
    raise ValueError(f"Unable to find target column {t} in {preds.keys()}")


def get_feature_name(model: LudwigModel, target: str) -> str:
    t = target.lower()
    for c in model.training_set_metadata.keys():
        if c.lower() == t:
            return c
    raise ValueError(f"Unable to find target column {t} in {model.training_set_metadata.keys()}")
