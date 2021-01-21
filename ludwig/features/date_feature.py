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
import logging
from datetime import date
from datetime import datetime

import numpy as np
import tensorflow as tf
from dateutil.parser import parse

from ludwig.constants import *
from ludwig.encoders.date_encoders import ENCODER_REGISTRY
from ludwig.features.base_feature import InputFeature
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)

DATE_VECTOR_LENGTH = 9


class DateFeatureMixin(object):
    type = DATE
    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': '',
        'datetime_format': None
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        return dataset_df

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def date_to_list(date_str, datetime_format, preprocessing_parameters):
        try:
            if datetime_format is not None:
                datetime_obj = datetime.strptime(date_str, datetime_format)
            else:
                datetime_obj = parse(date_str)
        except:
            logging.error(
                'Error parsing date: {}. '
                'Please provide a datetime format that parses it '
                'in the preprocessing section of the date feature '
                'in the config. '
                'The preprocessing fill in value will be used.'
                'For more details: '
                'https://ludwig.ai/user_guide/#date-features-preprocessing'
                    .format(date_str)
            )
            fill_value = preprocessing_parameters['fill_value']
            if fill_value != '':
                datetime_obj = parse(fill_value)
            else:
                datetime_obj = datetime.now()

        yearday = (
                datetime_obj.toordinal() -
                date(datetime_obj.year, 1, 1).toordinal() + 1
        )

        midnight = datetime_obj.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
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
            second_of_day
        ]

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        datetime_format = preprocessing_parameters['datetime_format']
        proc_df[feature[PROC_COLUMN]] = backend.df_engine.map_objects(
            input_df[feature[COLUMN]],
            lambda x: np.array(DateFeatureMixin.date_to_list(
                x, datetime_format, preprocessing_parameters
            ), dtype=np.int16)
        )
        return proc_df


class DateInputFeature(DateFeatureMixin, InputFeature):
    encoder = 'embed'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.int16

        inputs_encoded = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return inputs_encoded

    @classmethod
    def get_input_dtype(cls):
        return tf.int16

    def get_input_shape(self):
        return DATE_VECTOR_LENGTH,

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = ENCODER_REGISTRY
