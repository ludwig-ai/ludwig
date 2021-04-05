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
import sys
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.encoders.image_encoders import ENCODER_REGISTRY
from ludwig.features.base_feature import InputFeature
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.image_utils import greyscale
from ludwig.utils.image_utils import num_channels_in_image
from ludwig.utils.image_utils import resize_image
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)


class ImageFeatureMixin(object):
    type = IMAGE
    preprocessing_defaults = {
        'missing_value_strategy': BACKFILL,
        'in_memory': True,
        'resize_method': 'interpolate',
        'scaling': 'pixel_normalization',
        'num_processes': 1
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        return dataset_df

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {
            PREPROCESSING: preprocessing_parameters
        }

    @staticmethod
    def _read_image_and_resize(
            filepath,
            img_width,
            img_height,
            should_resize,
            num_channels,
            resize_method,
            user_specified_num_channels
    ):
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
        try:
            from skimage.io import imread
        except ImportError:
            logger.error(
                ' scikit-image is not installed. '
                'In order to install all image feature dependencies run '
                'pip install ludwig[image]'
            )
            sys.exit(-1)

        img = imread(filepath)
        img_num_channels = num_channels_in_image(img)
        if img_num_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1))

        if should_resize:
            img = resize_image(img, (img_height, img_width), resize_method)

        if user_specified_num_channels is True:

            # convert to greyscale if needed
            if num_channels == 1 and (
                    img_num_channels == 3 or img_num_channels == 4):
                img = greyscale(img)
                img_num_channels = 1

            # Number of channels is specified by the user
            img_padded = np.zeros((img_height, img_width, num_channels),
                                  dtype=np.uint8)
            min_num_channels = min(num_channels, img_num_channels)
            img_padded[:, :, :min_num_channels] = img[:, :, :min_num_channels]
            img = img_padded

            if img_num_channels != num_channels:
                logger.warning(
                    "Image {0} has {1} channels, where as {2} "
                    "channels are expected. Dropping/adding channels "
                    "with 0s as appropriate".format(
                        filepath, img_num_channels, num_channels))
        else:
            # If the image isn't like the first image, raise exception
            if img_num_channels != num_channels:
                raise ValueError(
                    'Image {0} has {1} channels, unlike the first image, which '
                    'has {2} channels. Make sure all the images have the same '
                    'number of channels or use the num_channels property in '
                    'image preprocessing'.format(filepath,
                                                 img_num_channels,
                                                 num_channels))

        if img.shape[0] != img_height or img.shape[1] != img_width:
            raise ValueError(
                "Images are not of the same size. "
                "Expected size is {0}, "
                "current image size is {1}."
                "Images are expected to be all of the same size "
                "or explicit image width and height are expected "
                "to be provided. "
                "Additional information: "
                "https://ludwig-ai.github.io/ludwig-docs/user_guide/#image-features-preprocessing"
                    .format([img_height, img_width, num_channels], img.shape)
            )

        return img

    @staticmethod
    def _finalize_preprocessing_parameters(
            preprocessing_parameters,
            first_image_path
    ):
        """
        Helper method to determine the height, width and number of channels for
        preprocessing the image data. This is achieved by looking at the
        parameters provided by the user. When there are some missing parameters,
        we fall back on to the first image in the dataset. The assumption being
        that all the images in the data are expected be of the same size with
        the same number of channels
        """
        # Read the first image in the dataset
        try:
            from skimage.io import imread
        except ImportError:
            logger.error(
                ' scikit-image is not installed. '
                'In order to install all image feature dependencies run '
                'pip install ludwig[image]'
            )
            sys.exit(-1)

        first_image = imread(first_image_path)
        first_img_height = first_image.shape[0]
        first_img_width = first_image.shape[1]
        first_img_num_channels = num_channels_in_image(first_image)

        should_resize = False
        if (HEIGHT in preprocessing_parameters or
                WIDTH in preprocessing_parameters):
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
        else:
            # User hasn't specified height and width.
            # So we assume that all images have the same width and height.
            # Thus the width and height of the first one are the same
            # as all the other ones
            height = first_img_height
            width = first_img_width

        if NUM_CHANNELS in preprocessing_parameters:
            # User specified num_channels in the model/feature config
            user_specified_num_channels = True
            num_channels = preprocessing_parameters[NUM_CHANNELS]
        else:
            user_specified_num_channels = False
            num_channels = first_img_num_channels

        assert isinstance(num_channels, int), ValueError(
            'Number of image channels needs to be an integer'
        )

        return (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            first_image
        )

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        in_memory = preprocessing_parameters['in_memory']
        if PREPROCESSING in feature and 'in_memory' in feature[PREPROCESSING]:
            in_memory = feature[PREPROCESSING]['in_memory']

        num_processes = preprocessing_parameters['num_processes']
        if PREPROCESSING in feature and 'num_processes' in feature[
            PREPROCESSING]:
            num_processes = feature[PREPROCESSING]['num_processes']

        src_path = None
        if hasattr(input_df, 'src'):
            src_path = os.path.dirname(os.path.abspath(input_df.src))

        num_images = len(input_df)
        if num_images == 0:
            raise ValueError('There are no images in the dataset provided.')

        first_path = next(iter(input_df[feature[COLUMN]]))

        if src_path is None and not os.path.isabs(first_path):
            raise ValueError('Image file paths must be absolute')

        first_path = get_abs_path(src_path, first_path)

        (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            first_image
        ) = ImageFeatureMixin._finalize_preprocessing_parameters(
            preprocessing_parameters, first_path
        )

        metadata[feature[NAME]][PREPROCESSING]['height'] = height
        metadata[feature[NAME]][PREPROCESSING]['width'] = width
        metadata[feature[NAME]][PREPROCESSING][
            'num_channels'] = num_channels

        read_image_and_resize = partial(
            ImageFeatureMixin._read_image_and_resize,
            img_width=width,
            img_height=height,
            should_resize=should_resize,
            num_channels=num_channels,
            resize_method=preprocessing_parameters['resize_method'],
            user_specified_num_channels=user_specified_num_channels
        )

        if in_memory:
            # Number of processes to run in parallel for preprocessing
            metadata[feature[NAME]][PREPROCESSING][
                'num_processes'] = num_processes
            metadata[feature[NAME]]['reshape'] = (height, width, num_channels)

            # Split the dataset into pools only if we have an explicit request to use
            # multiple processes. In case we have multiple input images use the
            # standard code anyway.
            if backend.supports_multiprocessing and (
                    num_processes > 1 or num_images > 1):
                all_file_paths = [get_abs_path(src_path, file_path)
                                  for file_path in input_df[feature[NAME]]]

                with Pool(num_processes) as pool:
                    logger.debug(
                        'Using {} processes for preprocessing images'.format(
                            num_processes
                        )
                    )
                    proc_df[feature[PROC_COLUMN]] = pool.map(read_image_and_resize, all_file_paths)
            else:
                # If we're not running multiple processes and we are only processing one
                # image just use this faster shortcut, bypassing multiprocessing.Pool.map
                logger.debug(
                    'No process pool initialized. Using internal process for preprocessing images'
                )

                proc_df[feature[PROC_COLUMN]] = backend.df_engine.map_objects(
                    input_df[feature[COLUMN]],
                    lambda file_path: read_image_and_resize(get_abs_path(src_path, file_path))
                )
        else:
            backend.check_lazy_load_supported(feature)

            all_file_paths = [get_abs_path(src_path, file_path)
                              for file_path in input_df[feature[NAME]]]

            data_fp = os.path.splitext(input_df.src)[0] + '.hdf5'
            mode = 'w'
            if os.path.isfile(data_fp):
                mode = 'r+'

            with h5py.File(data_fp, mode) as h5_file:
                # todo future add multiprocessing/multithreading
                image_dataset = h5_file.create_dataset(
                    feature[PROC_COLUMN] + '_data',
                    (num_images, height, width, num_channels),
                    dtype=np.uint8
                )
                for i, filepath in enumerate(all_file_paths):
                    image_dataset[i, :height, :width, :] = (
                        read_image_and_resize(filepath)
                    )
                h5_file.flush()

            proc_df[feature[PROC_COLUMN]] = np.arange(num_images)
        return proc_df


class ImageInputFeature(ImageFeatureMixin, InputFeature):
    height = 0
    width = 0
    num_channels = 0
    scaling = 'pixel_normalization'
    encoder = 'stacked_cnn'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.uint8

        # csting and rescaling
        inputs = tf.cast(inputs, tf.float32) / 255

        inputs_encoded = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return inputs_encoded

    @classmethod
    def get_input_dtype(cls):
        return tf.uint8

    def get_input_shape(self):
        return self.height, self.width, self.num_channels

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        for key in ['height', 'width', 'num_channels', 'scaling']:
            input_feature[key] = feature_metadata[PREPROCESSING][key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, PREPROCESSING, {})

    encoder_registry = ENCODER_REGISTRY


image_scaling_registry = {
    'pixel_normalization': lambda x: x * 1.0 / 255,
    'pixel_standardization': lambda x: tf.map_fn(
        lambda f: tf.image.per_image_standardization(f), x)
}
