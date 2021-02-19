#! /usr/bin/env python
# coding=utf-8
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
import pandas as pd

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import BINARY
from ludwig.features.feature_utils import SEQUENCE_TYPES
from ludwig.utils.data_utils import DICT_FORMATS, DATAFRAME_FORMATS, \
    normalize_numpy
from ludwig.utils.misc_utils import get_from_registry


def postprocess(
        predictions,
        output_features,
        training_set_metadata,
        output_directory='',
        backend=LOCAL_BACKEND,
        skip_save_unprocessed_output=False,
):
    if not backend.is_coordinator():
        # Only save unprocessed output on the coordinator
        skip_save_unprocessed_output = True

    postprocessed = {}
    for of_name, output_feature in output_features.items():
        postprocessed[of_name] = output_feature.postprocess_predictions(
            predictions[of_name],
            training_set_metadata[of_name],
            output_directory=output_directory,
            skip_save_unprocessed_output=skip_save_unprocessed_output
        )
    return postprocessed


def convert_predictions(predictions, output_features, training_set_metadata,
                        return_type='dict'):
    convert_fn = get_from_registry(
        return_type,
        conversion_registry
    )
    return convert_fn(
        predictions,
        output_features,
        training_set_metadata,
    )


def convert_to_dict(
        predictions,
        output_features,
        training_set_metadata,
):
    return predictions


def convert_to_df(
        predictions,
        output_features,
        training_set_metadata,
):
    data_for_df = {}
    for of_name, output_feature in output_features.items():
        output_feature_dict = predictions[of_name]
        for key_val in output_feature_dict.items():
            output_subgroup_name, output_type_value = key_val
            if (hasattr(output_type_value, 'shape') and
                len(output_type_value.shape)) > 1:
                if output_feature.type in SEQUENCE_TYPES:
                    data_for_df[
                        '{}_{}'.format(of_name, output_subgroup_name)
                    ] = output_type_value.tolist()
                else:
                    for i, value in enumerate(output_type_value.T):
                        if (of_name in training_set_metadata and
                                'idx2str' in training_set_metadata[of_name]):
                            class_name = training_set_metadata[of_name][
                                'idx2str'][i]
                        elif output_feature.type == BINARY:
                            if (of_name in training_set_metadata and
                                    'bool2str' in training_set_metadata[of_name]):
                                class_name = training_set_metadata[of_name][
                                    'bool2str'][i]
                            else:
                                class_name = 'True' if i == 1 else 'False'
                        else:
                            class_name = str(i)
                        data_for_df[
                            '{}_{}_{}'.format(
                                of_name,
                                output_subgroup_name,
                                class_name
                            )
                        ] = normalize_numpy(value)
            else:
                data_for_df[
                    '{}_{}'.format(
                        of_name,
                        output_subgroup_name
                    )
                ] = normalize_numpy(output_type_value)
    output_df = pd.DataFrame(data_for_df)
    return output_df


conversion_registry = {
    **{format: convert_to_dict for format in DICT_FORMATS},
    **{format: convert_to_df for format in DATAFRAME_FORMATS},
}
