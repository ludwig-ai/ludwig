import pandas as pd

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, PREPROCESSING, SPLIT


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
