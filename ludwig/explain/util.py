import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, PREPROCESSING, SPLIT
from ludwig.data.split import get_splitter
from ludwig.features.base_feature import BaseFeature
from ludwig.utils.torch_utils import copy_module_and_tie_weights


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


def replace_layer_with_copy(feat: BaseFeature, target_layer: torch.nn.Module):
    """Replaces a layer in a feature with a copy of the layer in-place.

    This is useful in a tied weights scenario, where a single encoder may be used by multiple features. If we leave
    as-is, Captum complains about the resulting computation graph. The solution is to create an identical
    (deep) copy of the layer fed into Captum: https://github.com/pytorch/captum/issues/794#issuecomment-1093021638

    This is safe to do during the explain step because we are essentially running inference, and no model artifacts are
    being saved during the explain step.

    TODO(geoffrey): if a user ever wants to train immediately after explain (i.e. w/o loading weights from the disk),
    we might want to implement this as a context so that we can restore the original encoder object at the end.
    Will defer this implementation for now because that scenario seems unlikely.

    This approach ensures that at most 2 copies of the encoder object are in memory at any given time.
    """
    with torch.no_grad():
        # Get the original encoder object, then deep copy the original encoder object and set the copy as this
        # feature's encoder object. We keep the copies of the `target_layer` parameters to enable explanations of
        # individual input features.
        copy_module_and_tie_weights(feat.encoder_obj, keep_copy=[target_layer])
