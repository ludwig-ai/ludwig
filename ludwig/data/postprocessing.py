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

from ludwig.features.feature_registries import output_type_registry
from ludwig.features.feature_utils import SEQUENCE_TYPES
from ludwig.utils.misc import get_from_registry


def postprocess_results(
        result,
        output_feature,
        metadata,
        experiment_dir_name='',
        skip_save_unprocessed_output=False,
):
    feature = get_from_registry(
        output_feature['type'], output_type_registry
    )
    return feature.postprocess_results(
        output_feature,
        result,
        metadata,
        experiment_dir_name,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
    )


def postprocess(
        results,
        output_features,
        metadata,
        experiment_dir_name='',
        skip_save_unprocessed_output=False,
):
    postprocessed = {}
    for output_feature in output_features:
        postprocessed[output_feature['name']] = postprocess_results(
            results[output_feature['name']],
            output_feature,
            metadata.get(output_feature['name'], {}),
            experiment_dir_name=experiment_dir_name,
            skip_save_unprocessed_output=skip_save_unprocessed_output,
        )
    return postprocessed


def postprocess_df(
    model_output,
    output_features,
    metadata,
    experiment_dir_name='',
        skip_save_unprocessed_output=True,
):
    postprocessed_output = postprocess(
        model_output,
        output_features,
        metadata,
        experiment_dir_name=experiment_dir_name,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
    )
    data_for_df = {}
    for output_feature in output_features:
        output_feature_name = output_feature['name']
        output_feature_type = output_feature['type']
        output_feature_dict = postprocessed_output[output_feature_name]
        for key_val in output_feature_dict.items():
            output_subgroup_name, output_type_value = key_val
            if (hasattr(output_type_value, 'shape') and
                len(output_type_value.shape)) > 1:
                if output_feature_type in SEQUENCE_TYPES:
                    data_for_df[
                        '{}_{}'.format(
                            output_feature_name,
                            output_subgroup_name
                        )
                    ] = output_type_value.tolist()
                else:
                    for i, value in enumerate(output_type_value.T):
                        if (output_feature_name in metadata and
                                'idx2str' in metadata[output_feature_name]):
                            class_name = metadata[output_feature_name][
                                'idx2str'][i]
                        else:
                            class_name = str(i)
                        data_for_df[
                            '{}_{}_{}'.format(
                                output_feature_name,
                                output_subgroup_name,
                                class_name
                            )
                        ] = value
            else:
                data_for_df[
                    '{}_{}'.format(
                        output_feature_name,
                        output_subgroup_name
                    )
                ] = output_type_value
    output_df = pd.DataFrame(data_for_df)
    return output_df
