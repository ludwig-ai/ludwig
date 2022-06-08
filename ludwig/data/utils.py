from typing import Any, Dict, Union

import dask.dataframe as dd
import numpy as np
from pandas import DataFrame


def convert_to_dict(
    predictions: Union[DataFrame, dd.core.DataFrame],
    output_features: Dict[str, Any],
):
    """Convert predictions from DataFrame format to a dictionary."""
    output = {}
    for of_name, output_feature in output_features.items():
        feature_keys = {k for k in predictions.columns if k.startswith(of_name)}
        feature_dict = {}
        for key in feature_keys:
            subgroup = key[len(of_name) + 1 :]

            values = predictions[key]
            try:
                values = np.stack(values.to_numpy())
            except ValueError:
                values = values.to_list()

            feature_dict[subgroup] = values
        output[of_name] = feature_dict
    return output
