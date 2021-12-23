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
from datetime import date, datetime
from typing import Any, Dict

import numpy as np
import torch
from dateutil.parser import parse

from ludwig.constants import COLUMN, DATE, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PROC_COLUMN, TIED
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)

DATE_VECTOR_LENGTH = 9


class DateFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return DATE

    @staticmethod
    def preprocessing_defaults():
        return {"missing_value_strategy": FILL_WITH_CONST, "fill_value": "", "datetime_format": None}

    @staticmethod
    def preprocessing_schema():
        return {
            "missing_value_strategy": {"type": "string", "enum": MISSING_VALUE_STRATEGY_OPTIONS},
            "fill_value": {"type": "string"},
            "computed_fill_value": {"type": "string"},
            "datetime_format": {"type": ["string", "null"]},
        }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {"preprocessing": preprocessing_parameters}

    @staticmethod
    def date_to_list(date_str, datetime_format, preprocessing_parameters):
        try:
            if datetime_format is not None:
                datetime_obj = datetime.strptime(date_str, datetime_format)
            else:
                datetime_obj = parse(date_str)
        except Exception as e:
            logging.error(
                f"Error parsing date: {date_str} with error {e} "
                "Please provide a datetime format that parses it "
                "in the preprocessing section of the date feature "
                "in the config. "
                "The preprocessing fill in value will be used."
                "For more details: "
                "https://ludwig.ai/user_guide/#date-features-preprocessing"
            )
            fill_value = preprocessing_parameters["fill_value"]
            if fill_value != "":
                datetime_obj = parse(fill_value)
            else:
                datetime_obj = datetime.now()

        yearday = datetime_obj.toordinal() - date(datetime_obj.year, 1, 1).toordinal() + 1

        midnight = datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        second_of_day = (datetime_obj - midnight).seconds

        return [
            datetime_obj.year,
            datetime_obj.month,
            datetime_obj.day,
            datetime_obj.weekday(),
            yearday,
            datetime_obj.hour,
            datetime_obj.minute,
            datetime_obj.second,
            second_of_day,
        ]

    def add_feature_data(
        feature_config: Dict[str, Any],
        input_df: DataFrame,
        proc_df: Dict[str, DataFrame],
        metadata: Dict[str, Any],
        preprocessing_parameters: Dict[str, Any],
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
    encoder = "embed"

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

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
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        pass

    def create_sample_input(self):
        return torch.Tensor([[2013, 2, 26, 1, 57, 0, 0, 0, 0], [2015, 2, 26, 1, 57, 0, 0, 0, 0]]).type(torch.int32)

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
