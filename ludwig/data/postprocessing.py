#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch

from ludwig.backend import LOCAL_BACKEND
from ludwig.utils.data_utils import DATAFRAME_FORMATS, DICT_FORMATS, to_numpy_dataset
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.strings_utils import make_safe_filename


def postprocess(
    predictions,
    output_features,
    training_set_metadata,
    output_directory="",
    backend=LOCAL_BACKEND,
    skip_save_unprocessed_output=False,
):
    if not backend.is_coordinator():
        # Only save unprocessed output on the coordinator
        skip_save_unprocessed_output = True

    saved_keys = set()
    if not skip_save_unprocessed_output:
        _save_as_numpy(predictions, output_directory, saved_keys)

    for of_name, output_feature in output_features.items():
        predictions = output_feature.postprocess_predictions(
            predictions,
            training_set_metadata[of_name],
            output_directory=output_directory,
            backend=backend,
        )

    # Save any new columns but do not save the original columns again
    if not skip_save_unprocessed_output:
        _save_as_numpy(predictions, output_directory, saved_keys)

    return predictions


def _save_as_numpy(predictions, output_directory, saved_keys):
    predictions = predictions[[c for c in predictions.columns if c not in saved_keys]]
    npy_filename = os.path.join(output_directory, "{}.npy")
    numpy_predictions = to_numpy_dataset(predictions)
    for k, v in numpy_predictions.items():
        k = k.replace("<", "[").replace(">", "]")  # Replace <UNK> and <PAD> with [UNK], [PAD]
        if k not in saved_keys:
            np.save(npy_filename.format(make_safe_filename(k)), v)
            saved_keys.add(k)


def convert_dict_to_df(predictions: Dict[str, Dict[str, Union[List[Any], torch.Tensor, np.array]]]) -> pd.DataFrame:
    """Converts a dictionary of predictions into a pandas DataFrame.

    Example format of predictions dictionary:

    {
        "binary_C82EB": {
            "predictions": torch.tensor([True, True, True, False]),
            "probabilities": torch.tensor([[0.4777, 0.5223], [0.4482, 0.5518], [0.4380, 0.5620], [0.5059, 0.4941]]),
        },
        "category_1491D": {
            "predictions": ["NkNUG", "NkNUG", "NkNUG", "NkNUG"],
            "probabilities": torch.tensor(
                [
                    [0.1058, 0.4366, 0.1939, 0.2637],
                    [0.0816, 0.4807, 0.1978, 0.2399],
                    [0.0907, 0.4957, 0.1829, 0.2308],
                    [0.0728, 0.5015, 0.1900, 0.2357],
                ]
            ),
        },
        "num_7B25F": {"predictions": torch.tensor([2.0436, 2.1158, 2.1222, 2.1964])},
    }
    """
    output = {}
    for of_name, preds_dict in predictions.items():
        for key, value in preds_dict.items():
            output_key = f"{of_name}_{key}"
            if not isinstance(value, list):
                value = value.tolist()
            output[output_key] = value
    return pd.DataFrame.from_dict(output)


def convert_predictions(predictions, output_features, return_type="dict"):
    convert_fn = get_from_registry(return_type, conversion_registry)
    return convert_fn(
        predictions,
        output_features,
    )


def convert_to_dict(
    predictions,
    output_features,
):
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


def convert_to_df(
    predictions,
    output_features,
):
    return predictions


conversion_registry = {
    **{format: convert_to_dict for format in DICT_FORMATS},
    **{format: convert_to_df for format in DATAFRAME_FORMATS},
}
