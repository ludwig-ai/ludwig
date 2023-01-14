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
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from dateutil.parser import parse

from ludwig.constants import COLUMN, DATE, PROC_COLUMN
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature
from ludwig.schema.features.date_feature import DateInputFeatureConfig
from ludwig.types import FeatureConfigDict, FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.date_utils import create_vector_from_datetime_obj
from ludwig.utils.types import DataFrame, TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)

DATE_VECTOR_LENGTH = 9


class _DatePreprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if torch.jit.isinstance(v, List[torch.Tensor]):
            v = torch.stack(v)

        if torch.jit.isinstance(v, torch.Tensor):
            return v.to(dtype=torch.int)
        else:
            raise ValueError(f"Unsupported input: {v}")


class DateFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return DATE

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        return {"preprocessing": preprocessing_parameters}

    @staticmethod
    def date_to_list(date_str, datetime_format, preprocessing_parameters):
        try:
            if isinstance(date_str, datetime):
                datetime_obj = date_str
            elif datetime_format is not None:
                datetime_obj = datetime.strptime(date_str, datetime_format)
            else:
                datetime_obj = parse(date_str)
        except Exception as e:
            logger.error(
                f"Error parsing date: '{date_str}' with error '{e}' "
                "Please provide a datetime format that parses it "
                "in the preprocessing section of the date feature "
                "in the config. "
                "The preprocessing fill in value will be used."
                "For more details: "
                "https://ludwig-ai.github.io/ludwig-docs/latest/configuration/features/date_features/#date-features-preprocessing"  # noqa
            )
            fill_value = preprocessing_parameters["fill_value"]
            if fill_value != "":
                datetime_obj = parse(fill_value)
            else:
                datetime_obj = datetime.now()

        return create_vector_from_datetime_obj(datetime_obj)

    @staticmethod
    def add_feature_data(
        feature_config: FeatureConfigDict,
        input_df: DataFrame,
        proc_df: Dict[str, DataFrame],
        metadata: TrainingSetMetadataDict,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,  # Union[Backend, str]
        skip_save_processed_input: bool,
    ) -> None:
        datetime_format = preprocessing_parameters["datetime_format"]
        proc_df[feature_config[PROC_COLUMN]] = backend.df_engine.map_objects(
            input_df[feature_config[COLUMN]],
            lambda x: np.array(
                DateFeatureMixin.date_to_list(x, datetime_format, preprocessing_parameters), dtype=np.int16
            ),
        )
        return proc_df


class DateInputFeature(DateFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: DateInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.int16, torch.int32, torch.int64]
        inputs_encoded = self.encoder_obj(inputs)
        return inputs_encoded

    @property
    def input_dtype(self):
        return torch.int16

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([DATE_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    def create_sample_input(self, batch_size: int = 2):
        date = [2013, 2, 26, 1, 57, 0, 0, 0, 0]
        return torch.Tensor([date for _ in range(batch_size)]).type(torch.int32)

    @staticmethod
    def get_schema_cls():
        return DateInputFeatureConfig

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _DatePreprocessing(metadata)
