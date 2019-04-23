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
from ludwig.utils.image_utils import get_abs_path
from ludwig.utils.image_utils import resize_image
from ludwig.utils.image_utils import num_channels_in_image
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
    def _read_image_and_resize(filepath,
                               img_width,
                               img_height,
                               should_resize,
                               num_channels,
                               resize_method,
                               user_specified_num_channels):
        """
        :param filepath: path to the image
        :param img_width: expected width of the image
        :param img_height: expected height of the image
        :param should_resize: Should the image be resized?
        :param resize_method: type of resizing method
        :param num_channels: expected number of channels in the first image
        :param user_specified_num_channels: did the user specify num channels?
        :return: image object

        Helper method to read and resize an image according to model defn.
        If the user doesn't specify a number of channels, we use the first image
        in the dataset as the source of truth. If any image in the dataset
        doesn't have the same number of channels as the first image,
        raise an exception.

        If the user specifies a number of channels, we try to convert all the
        images to the specifications by dropping channels/padding 0 channels
        """

        img = imread(filepath)
        img_num_channels = num_channels_in_image(img)
        if img_num_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1))

        if should_resize:
            img = resize_image(img, (img_height, img_width), resize_method)

        if user_specified_num_channels is True:
            # Number of channels is specified by the user
            img_padded = np.zeros((img_height, img_width, num_channels))
            min_num_channels = min(num_channels, img_num_channels)
            img_padded[:,:,:min_num_channels] = img[:,:,:min_num_channels]
            img = img_padded

            if img_num_channels != num_channels:
                logging.warning(
                    "Image {0} has {1} channels, where as {2}"
                    " channels are expected. Dropping/adding channels"
                    "with 0s as appropriate".format(
                        filepath, img_num_channels, num_channels))
        else:
            # If the image isn't like the first image, raise exception
            if img_num_channels != num_channels:
                raise ValueError(
                    'Image {0} has {1} channels, unlike the first image, which'
                    ' has {2} channels'.format(filepath,
                                               img_num_channels,
                                               num_channels))
        return img

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        set_default_value(
            feature['preprocessing'],
            'in_memory',
            preprocessing_parameters['in_memory']
        )

        csv_path = None
        if hasattr(dataset_df, 'csv'):
            csv_path = os.path.dirname(os.path.abspath(dataset_df.csv))

        num_images = len(dataset_df)
        if num_images == 0:
            raise ValueError('There are no images in the dataset provided.')

        height = 0
        width = 0
        should_resize = False
        if ('height' in preprocessing_parameters or
                'width' in preprocessing_parameters):
            should_resize = True
            try:
                height = int(preprocessing_parameters[HEIGHT])
                width = int(preprocessing_parameters[WIDTH])
            except ValueError as e:
                raise ValueError(
                    'Image height and width must be set and have '
                    'positive integer values: ' + str(e)
                )
            if height <= 0 or width <= 0:
                raise ValueError(
                    'Image height and width must be positive integers'
                )

        # here if a width and height have not been specified
        # we assume that all images have the same width and height
        # thus the width and height of the first one are the same
        # of all the other ones
        if (csv_path is None and
                not os.path.isabs(dataset_df[feature['name']][0])):
            raise ValueError(
                'Image file paths must be absolute'
            )

        first_image = imread(
            get_abs_path(
                csv_path,
                dataset_df[feature['name']][0]
            )
        )

        first_img_height = first_image.shape[0]
        first_img_width = first_image.shape[1]
        first_img_num_channels = num_channels_in_image(first_image)

        if height == 0 or width == 0:
            # User hasn't specified height and width
            height = first_img_height
            width = first_img_width

        # User specified num_channels in the model/feature definition
        user_specified_num_channels = False
        num_channels = first_img_num_channels
        if NUM_CHANNELS in preprocessing_parameters:
            user_specified_num_channels = True
            num_channels = preprocessing_parameters[NUM_CHANNELS]

        assert isinstance(num_channels, int), ValueError(
            'Number of image channels needs to be an integer')

        metadata[feature['name']]['preprocessing']['height'] = height
        metadata[feature['name']]['preprocessing']['width'] = width
        metadata[feature['name']]['preprocessing'][
            'num_channels'] = num_channels

        if feature['preprocessing']['in_memory']:
            data[feature['name']] = np.empty(
                (num_images, height, width, num_channels),
                dtype=np.int8
            )
            for i in range(len(dataset_df)):
                filepath = get_abs_path(
                        csv_path,
                        dataset_df[feature['name']][i]
                )

                img = ImageBaseFeature._read_image_and_resize(
                    filepath,
                    width,
                    height,
                    should_resize,
                    num_channels,
                    preprocessing_parameters['resize_method'],
                    user_specified_num_channels
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
                    filepath = get_abs_path(
                            csv_path,
                            dataset_df[feature['name']][i]
                    )

                    img = ImageBaseFeature._read_image_and_resize(
                        filepath,
                        width,
                        height,
                        should_resize,
                        num_channels,
                        preprocessing_parameters['resize_method'],
                        user_specified_num_channels
                    )

                    image_dataset[i, :height, :width, :] = img

            data[feature['name']] = np.arange(num_images)


class ImageInputFeature(ImageBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.height = 0
        self.width = 0
        self.num_channels = 0

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

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'preprocessing', {})


image_encoder_registry = {
    'stacked_cnn': Stacked2DCNN,
    'resnet': ResNetEncoder
}
