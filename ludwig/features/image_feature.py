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
from skimage.io import imread

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
        'missing_value_strategy': BACKFILL,
        'in_memory': True,
        'resize_method': 'crop_or_pad'
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        set_default_value(
            feature,
            'in_memory',
            preprocessing_parameters['in_memory']
        )

        if ('height' in preprocessing_parameters or
                'width' in preprocessing_parameters):
            should_resize = True
            try:
                provided_height = int(preprocessing_parameters[HEIGHT])
                provided_width = int(preprocessing_parameters[WIDTH])
            except ValueError as e:
                raise ValueError(
                    'Image height and width must be set and have '
                    'positive integer values: ' + str(e)
                )
            if (provided_height <= 0 or provided_width <= 0):
                raise ValueError(
                    'Image height and width must be positive integers'
                )
        else:
            should_resize = False

        csv_path = os.path.dirname(os.path.abspath(dataset_df.csv))

        num_images = len(dataset_df)

        height = 0
        width = 0
        num_channels = 1

        if num_images > 0:
            # here if a width and height have not been specified
            # we assume that all images have the same wifth and im_height
            # thus the width and height of the first one are the same
            # of all the other ones
            first_image = imread(
                os.path.join(csv_path, dataset_df[feature['name']][0])
            )
            height = first_image.shape[0]
            width = first_image.shape[1]

            if first_image.ndim == 2:
                num_channels = 1
            else:
                num_channels = first_image.shape[2]

        if should_resize:
            height = provided_height
            width = provided_width

        metadata[feature['name']]['preprocessing']['height'] = height
        metadata[feature['name']]['preprocessing']['width'] = width
        metadata[feature['name']]['preprocessing'][
            'num_channels'] = num_channels

        if feature['in_memory']:
            data[feature['name']] = np.empty(
                (num_images, height, width, num_channels),
                dtype=np.int8
            )
            for i in range(len(dataset_df)):
                filename = os.path.join(
                    csv_path,
                    dataset_df[feature['name']][i]
                )
                img = imread(filename)
                if img.ndim == 2:
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                if should_resize:
                    img = resize_image(
                        img,
                        (height, width),
                        preprocessing_parameters['resize_method']
                    )
                data[feature['name']][i, :, :, :] = img
        else:
            data_fp = os.path.splitext(dataset_df.csv)[0] + '.hdf5'
            mode = 'w'
            if os.path.isfile(data_fp):
                mode = 'r+'
            with h5py.File(data_fp, mode) as h5_file:
                image_dataset = h5_file.create_dataset(
                    feature['name'] + '_data',
                    (num_images, height, width, num_channels),
                    dtype=np.uint8
                )
                for i in range(len(dataset_df)):
                    filename = os.path.join(
                        csv_path,
                        dataset_df[feature['name']][i]
                    )
                    img = imread(filename)
                    if img.ndim == 2:
                        img = img.reshape((img.shape[0], img.shape[1], 1))
                    if should_resize:
                        img = resize_image(
                            img,
                            (height, width),
                            preprocessing_parameters['resize_method'],
                        )

                    image_dataset[i, :height, :width, :] = img

            data[feature['name']] = np.arange(num_images)


class ImageInputFeature(ImageBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

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
            input_feature[dim] = feature_metadata['preprocessing'][dim]
        input_feature['data_hdf5_fp'] = (
            kwargs['model_definition']['data_hdf5_fp']
        )

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)


image_encoder_registry = {
    'stacked_cnn': Stacked2DCNN,
    'resnet': ResNetEncoder
}
