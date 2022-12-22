from typing import Any, Dict, Optional

import numpy as np

from ludwig.constants import SPLIT, TYPE
from ludwig.encoders.registry import get_encoder_cls
from ludwig.types import FeatureConfigDict, PreprocessingConfigDict
from ludwig.utils.dataframe_utils import is_dask_series_or_df
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.types import DataFrame


def convert_to_dict(
    predictions: DataFrame,
    output_features: Dict[str, FeatureConfigDict],
    backend: Optional["Backend"] = None,  # noqa: F821
):
    """Convert predictions from DataFrame format to a dictionary."""
    output = {}
    for of_name, output_feature in output_features.items():
        feature_keys = {k for k in predictions.columns if k.startswith(of_name)}
        feature_dict = {}
        for key in feature_keys:
            subgroup = key[len(of_name) + 1 :]

            values = predictions[key]
            if is_dask_series_or_df(values, backend):
                values = values.compute()
            try:
                values = np.stack(values.to_numpy())
            except ValueError:
                values = values.to_list()

            feature_dict[subgroup] = values
        output[of_name] = feature_dict
    return output


def set_fixed_split(preprocessing_params: PreprocessingConfigDict) -> PreprocessingConfigDict:
    """Sets the split policy explicitly to a fixed split.

    This potentially overrides the split configuration that the user set or what came from schema defaults.
    """

    return {
        **preprocessing_params,
        "split": {
            "type": "fixed",
            "column": SPLIT,
        },
    }


def merge_fixed_preprocessing_params(
    feature_type: str, preprocessing_params: Dict[str, Any], encoder_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Update preprocessing parameters if encoders require fixed preprocessing parameters."""
    encoder_type = encoder_params.get(TYPE)
    if encoder_type is None:
        return preprocessing_params

    encoder_class = get_encoder_cls(feature_type, encoder_type)
    return merge_dict(preprocessing_params, encoder_class.get_fixed_preprocessing_params(encoder_params))
