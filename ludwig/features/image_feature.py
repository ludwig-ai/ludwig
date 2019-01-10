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
import os

import h5py
import numpy as np
import tensorflow as tf
from imageio import imread

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.models.modules.image_encoders import ResNetEncoder
from ludwig.models.modules.image_encoders import Stacked2DCNN
from ludwig.utils.image_utils import resize_image
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value


class ImageBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = IMAGE

    preprocessing_defaults = {
        'missing_value_strategy': BACKFILL
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {}

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        an_image = imread(dataset_df[feature['name']][0])
        im_height = an_image.shape[0]
        im_width = an_image.shape[1]

        if an_image.ndim == 2:
            num_channels = 1
        else:
            num_channels = an_image.shape[2]

        num_images = len(dataset_df)

        if feature['should_resize']:
            im_height = feature[HEIGHT]
            im_width = feature[WIDTH]

        metadata[feature['name']] = {
            'height': im_height,
            'width': im_width,
            'num_channels': num_channels,
            'in_memory': feature['in_memory']
        }

        if feature['in_memory']:
            data[feature['name']] = np.empty(
                (num_images, im_height, im_width, num_channels),
                dtype=np.int8
            )
            for i in range(len(dataset_df)):
                filename = dataset_df[feature['name']][i]
                img = imread(filename)
                if img.ndim == 2:
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                if feature['should_resize']:
                    img = resize_image(
                        img,
                        (im_height, im_width),
                        feature['resize_method']
                    )
                data[feature['name']][i, :, :, :] = img
        else:
            data_fp = dataset_df.csv.replace('csv', 'hdf5')
            mode = 'w'
            if os.path.isfile(data_fp):
                mode = 'r+'
            with h5py.File(data_fp, mode) as h5_file:
                image_dataset = h5_file.create_dataset(
                    feature['name'] + '_data',
                    (num_images, im_height, im_width, num_channels),
                    dtype=np.uint8
                )
                for i in range(len(dataset_df)):
                    filename = dataset_df[feature['name']][i]
                    img = imread(filename)
                    if img.ndim == 2:
                        img = img.reshape((img.shape[0], img.shape[1], 1))
                    if feature['should_resize']:
                        img = resize_image(
                            img,
                            (im_height, im_width),
                            feature['resize_method'],
                        )

                    image_dataset[i, :im_height, :im_width, :] = img

            data[feature['name']] = np.arange(num_images)


class ImageInputFeature(ImageBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.should_resize = False

        self.height = 0
        self.width = 0
        self.num_channels = 0

        self.in_memory = True
        self.data_hdf5_fp = ''

        self.encoder = 'stacked_cnn'

        encoder_parameters = self.overwrite_defaults(feature)

        self.encoder_obj = self.get_image_encoder(encoder_parameters)

    def get_image_encoder(self, encoder_parameters):
        return get_from_registry(
            self.encoder, image_encoder_registry)(
            **encoder_parameters
        )

    def _get_input_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.placeholder(
            tf.float32,
            shape=[None, self.height, self.width, self.num_channels],
            name=self.name,
        )

    def build_input(
            self,
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        placeholder = self._get_input_placeholder()
        logging.debug('  targets_placeholder: {0}'.format(placeholder))

        feature_representation, feature_representation_size = self.encoder_obj(
            placeholder,
            regularizer,
            dropout_rate,
            is_training,
        )
        logging.debug(
            '  feature_representation: {0}'.format(feature_representation)
        )

        feature_representation = {
            'name': self.name,
            'type': self.type,
            'representation': feature_representation,
            'size': feature_representation_size,
            'placeholder': placeholder
        }
        return feature_representation

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        for dim in ['height', 'width', 'num_channels']:
            input_feature[dim] = feature_metadata[dim]
        input_feature['data_hdf5_fp'] = (
            kwargs['model_definition']['data_hdf5_fp']
        )

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'in_memory', True)

        if 'height' in input_feature or 'width' in input_feature:
            input_feature['should_resize'] = True
            try:
                input_feature[HEIGHT] = int(input_feature[HEIGHT])
                input_feature[WIDTH] = int(input_feature[WIDTH])
            except ValueError as e:
                raise ValueError(
                    'Image height and width must be set and have '
                    'positive integer values: ' + str(e)
                )
            if input_feature[HEIGHT] <= 0 or input_feature[WIDTH] <= 0:
                raise ValueError(
                    'Image height and width must be positive integers'
                )
            input_feature['should_resize'] = True
            set_default_value(input_feature, 'resize_method', 'crop_or_pad')
        else:
            input_feature['should_resize'] = False

        set_default_value(input_feature, 'tied_weights', None)


image_encoder_registry = {
    'stacked_cnn': Stacked2DCNN,
    'rednet': ResNetEncoder
}
