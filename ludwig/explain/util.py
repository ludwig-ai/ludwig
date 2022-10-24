import pandas as pd

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, PREPROCESSING, SPLIT
from ludwig.data.split import get_splitter


def filter_cols(df, cols):
    cols = {c.lower() for c in cols}
    retain_cols = [c for c in df.columns if c.lower() in cols]
    return df[retain_cols]


def prepare_data(model: LudwigModel, inputs_df: pd.DataFrame, sample_df: pd.DataFrame, target: str):
    config = model.config
    feature_cols = [feature[COLUMN] for feature in config[INPUT_FEATURES]]
    if SPLIT in config.get(PREPROCESSING, {}):
        # Keep columns required for Ludwig preprocessing
        splitter = get_splitter(**config[PREPROCESSING][SPLIT])
        feature_cols += splitter.required_columns
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
