from copy import deepcopy

import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, PREPROCESSING, SPLIT
from ludwig.data.split import get_splitter
from ludwig.features.base_feature import BaseFeature


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


def get_absolute_module_key_from_submodule(module: torch.nn.Module, submodule: torch.nn.Module):
    """Get the absolute module key for each param in the target layer.

    Assumes that the keys in the submodule are relative to the module.

    We find the params from the submodule in the module by comparing the data
    pointers, since the data returned by named_parameters is by reference.
    More information on checking if tensors point to the same place in storage can be found here:
    https://discuss.pytorch.org/t/any-way-to-check-if-two-tensors-have-the-same-base/44310/2
    """
    absolute_keys = []
    for module_key, module_param in module.named_parameters():
        for _, submodule_param in submodule.named_parameters():
            if submodule_param.data_ptr() == module_param.data_ptr():
                absolute_keys.append(module_key)
                break
    return absolute_keys


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

    At a high-level the approach is the following:
    1. Create a deep-copy of the entire encoder object and set it as the feature's encoder object
    2. Replace the tensors in the copied encoder object with the tensors from the original encoder object, except for
         the tensors in the target layer. We want to explain these tensors, so we want to keep them as deep copies.

    This approach ensures that at most 2 copies of the encoder object are in memory at any given time.
    """
    with torch.no_grad():
        # Get the original encoder object and a mapping from param names to the params themselves.
        orig_encoder_obj = feat.encoder_obj
        orig_encoder_obj_state_dict = orig_encoder_obj.state_dict()

        # Deep copy the original encoder object and set the copy as this feature's encoder object.
        copy_encoder_obj = deepcopy(orig_encoder_obj)
        feat.encoder_obj = copy_encoder_obj

        # We have to get the absolute module key in order to do string matching because the target_layer keys are
        # relative to itself. If we were to leave it as-is and attempt to suffix match, we may get duplicates for
        # common layers i.e. "LayerNorm.weight" and "LayerNorm.bias". Getting the absolute module key ensures we
        # use values like "transformer.module.embedding.LayerNorm.weight" instead.
        keys_to_keep_copy = get_absolute_module_key_from_submodule(orig_encoder_obj, target_layer)

        # Get the tensors to keep from the copied encoder object. These are the tensors in the target layer.
        for key, param in copy_encoder_obj.named_parameters():
            if key not in keys_to_keep_copy:
                param.data = orig_encoder_obj_state_dict[key].data
