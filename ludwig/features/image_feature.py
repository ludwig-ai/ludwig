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
from typing import Union

import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.encoders.image_encoders import ENCODER_REGISTRY
from ludwig.features.base_feature import InputFeature
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import upload_h5
from ludwig.utils.image_utils import greyscale
from ludwig.utils.image_utils import num_channels_in_image
from ludwig.utils.image_utils import resize_image
from ludwig.utils.image_utils import get_image_from_path, read_image
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)


image_scaling_registry = {
    'pixel_normalization': lambda x: x * 1.0 / 255,
    'pixel_standardization': lambda x: tf.map_fn(
        lambda f: tf.image.per_image_standardization(f), x)
}


class ImageFeatureMixin:
    type = IMAGE
    preprocessing_defaults = {
        'missing_value_strategy': BACKFILL,
        'in_memory': True,
        'resize_method': 'interpolate',
        'scaling': 'pixel_normalization',
        'num_processes': 1,
        'infer_image_num_channels': True,
        'infer_image_dimensions': True,
        'infer_image_max_height': 256,
        'infer_image_max_width': 256,
        'infer_image_sample_size': 100
    }

    preprocessing_schema = {
        'missing_value_strategy': {'type': 'string', 'enum': MISSING_VALUE_STRATEGY_OPTIONS},
        'in_memory': {'type': 'boolean'},
        'resize_method': {'type': 'string', 'enum': RESIZE_METHODS},
        'scaling': {'type': 'string', 'enum': list(image_scaling_registry.keys())},
        'num_processes': {'type': 'integer', 'minimum': 0},
        'height': {'type': 'integer', 'minimum': 0},
        'width': {'type': 'integer', 'minimum': 0},
        'num_channels': {'type': 'integer', 'minimum': 0},
        'infer_image_num_channels': {'type': 'boolean'},
        'infer_image_dimensions': {'type': 'boolean'},
        'infer_image_max_height': {'type': 'integer', 'minimum': 0},
        'infer_image_max_width': {'type': 'integer', 'minimum': 0},
        'infer_image_sample_size': {'type': 'integer', 'minimum': 0}
    }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {
            PREPROCESSING: preprocessing_parameters
        }

    @staticmethod
    def _read_image_and_resize(
            img_entry: Union[str, 'numpy.array'],
            img_width: int,
            img_height: int,
            should_resize: bool,
            num_channels: int,
            resize_method: str,
            user_specified_num_channels: int
    ):
        """
        :param img_entry Union[str, 'numpy.array']: if str file path to the
                image else numpy.array of the image itself
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
        img = read_image(img_entry)
        if img is None:
            logger.info(f"{img_entry} cannot be read")
            return None
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
                    "Image has {0} channels, where as {1} "
                    "channels are expected. Dropping/adding channels "
                    "with 0s as appropriate".format(
                        img_num_channels, num_channels))
        else:
            # If the image isn't like the first image, raise exception
            if img_num_channels != num_channels:
                raise ValueError(
                    'Image has {0} channels, unlike the first image, which '
                    'has {1} channels. Make sure all the images have the same '
                    'number of channels or use the num_channels property in '
                    'image preprocessing'.format(img_num_channels,
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
            preprocessing_parameters: dict,
            first_img_entry: Union[str, 'numpy.array'],
            src_path: str,
            input_feature_col: np.array
    ):
        """
        Helper method to determine the height, width and number of channels for
        preprocessing the image data. This is achieved by looking at the
        parameters provided by the user. When there are some missing parameters,
        we fall back on to the first image in the dataset. The assumption being
        that all the images in the data are expected be of the same size with
        the same number of channels
        """
        first_image = read_image(first_img_entry)

        inferred_sample = None
        if preprocessing_parameters[INFER_IMAGE_DIMENSIONS] or preprocessing_parameters[INFER_IMAGE_NUM_CHANNELS]:
            sample_size = min(len(input_feature_col), preprocessing_parameters[INFER_IMAGE_SAMPLE_SIZE])
            sample = [read_image(get_image_from_path(src_path, img)) for img in input_feature_col.iloc[:sample_size]]
            inferred_sample = [img for img in sample if img is not None]
            if len(inferred_sample) == 0:
                raise ValueError("No readable images in sample, "
                                 "image dimensions and/or num channels cannot be inferred")

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
            # Default to first image, or infer from sample.
            if preprocessing_parameters[INFER_IMAGE_DIMENSIONS]:
                should_resize = True

                height_avg = min(
                    sum(x.shape[0] for x in inferred_sample) / len(inferred_sample),
                    preprocessing_parameters[INFER_IMAGE_MAX_HEIGHT])
                width_avg = min(
                    sum(x.shape[1] for x in inferred_sample) / len(inferred_sample),
                    preprocessing_parameters[INFER_IMAGE_MAX_WIDTH])

                height, width = round(height_avg), round(width_avg)
                logger.debug("Inferring height: {0} and width: {1}".format(height, width))
            elif first_image is not None:
                height, width = first_image.shape[0], first_image.shape[1]
            else:
                raise ValueError("First image cannot be read, explicit image dimensions are not set, and"
                                 " infer_image_dimensions is false, so image dimensions are unknown")

        if NUM_CHANNELS in preprocessing_parameters:
            # User specified num_channels in the model/feature config
            user_specified_num_channels = True
            num_channels = preprocessing_parameters[NUM_CHANNELS]
        else:
            user_specified_num_channels = False
            if preprocessing_parameters[INFER_IMAGE_NUM_CHANNELS]:
                user_specified_num_channels = True
                num_channels = round(sum(num_channels_in_image(x) for x in inferred_sample) / len(inferred_sample))
            elif first_image is not None:
                num_channels = num_channels_in_image(first_image)
            else:
                raise ValueError("First image cannot be read, explicit image num channels is not set, and"
                                 " infer_image_num_channels is false, so image num channels is unknown")

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
            backend,
            skip_save_processed_input
    ):
        in_memory = preprocessing_parameters['in_memory']
        if PREPROCESSING in feature and 'in_memory' in feature[PREPROCESSING]:
            in_memory = feature[PREPROCESSING]['in_memory']

        num_processes = preprocessing_parameters['num_processes']
        if PREPROCESSING in feature and 'num_processes' in feature[
            PREPROCESSING]:
            num_processes = feature[PREPROCESSING]['num_processes']

        src_path = None
        if SRC in metadata:
            src_path = os.path.dirname(os.path.abspath(metadata.get(SRC)))

        num_images = len(input_df[feature[COLUMN]])
        if num_images == 0:
            raise ValueError('There are no images in the dataset provided.')

        first_img_entry = next(iter(input_df[feature[COLUMN]]))
        logger.debug(
            'Detected image feature type is {}'.format(type(first_img_entry))
        )

        if not isinstance(first_img_entry, str) \
                and not isinstance(first_img_entry, np.ndarray):
            raise ValueError(
                'Invalid image feature data type.  Detected type is {}, '
                'expect either string for file path or numpy array.'
                    .format(type(first_img_entry))
            )

        first_img_entry = get_image_from_path(src_path, first_img_entry)

        (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            first_image
        ) = ImageFeatureMixin._finalize_preprocessing_parameters(
            preprocessing_parameters, first_img_entry, src_path, input_df[feature[COLUMN]]
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

        # check to see if the active backend can support lazy loading of
        # image features from the hdf5 cache.
        backend.check_lazy_load_supported(feature)

        if in_memory or skip_save_processed_input:
            # Number of processes to run in parallel for preprocessing
            metadata[feature[NAME]][PREPROCESSING][
                'num_processes'] = num_processes
            metadata[feature[NAME]]['reshape'] = (height, width, num_channels)

            # Split the dataset into pools only if we have an explicit request to use
            # multiple processes. In case we have multiple input images use the
            # standard code anyway.
            if backend.supports_multiprocessing and (
                    num_processes > 1 or num_images > 1):
                all_img_entries = [get_abs_path(src_path, img_entry)
                                   if isinstance(img_entry, str) else img_entry
                                   for img_entry in input_df[feature[COLUMN]]]

                with Pool(num_processes) as pool:
                    logger.debug(
                        'Using {} processes for preprocessing images'.format(
                            num_processes
                        )
                    )
                    proc_df[feature[PROC_COLUMN]] = pool.map(
                        read_image_and_resize, all_img_entries
                    )
            else:
                # If we're not running multiple processes and we are only processing one
                # image just use this faster shortcut, bypassing multiprocessing.Pool.map
                logger.debug(
                    'No process pool initialized. Using internal process for preprocessing images'
                )

                # helper function for handling single image
                def _get_processed_image(img_store):
                    if isinstance(img_store, str):
                        return read_image_and_resize(
                            get_abs_path(src_path, img_store)
                        )
                    else:
                        return read_image_and_resize(img_store)

                proc_df[feature[PROC_COLUMN]] = backend.df_engine.map_objects(
                    input_df[feature[COLUMN]],
                    _get_processed_image
                )
        else:

            all_img_entries = [get_abs_path(src_path, img_entry)
                               if isinstance(img_entry, str) else img_entry
                               for img_entry in input_df[feature[COLUMN]]]

            data_fp = backend.cache.get_cache_path(
                metadata.get(SRC), metadata.get(CHECKSUM), TRAINING
            )
            with upload_h5(data_fp) as h5_file:
                # todo future add multiprocessing/multithreading
                image_dataset = h5_file.create_dataset(
                    feature[PROC_COLUMN] + '_data',
                    (num_images, height, width, num_channels),
                    dtype=np.uint8
                )
                for i, img_entry in enumerate(all_img_entries):
                    image_dataset[i, :height, :width, :] = (
                        read_image_and_resize(img_entry)
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
        assert inputs.dtype in [tf.uint8, tf.int64]

        # casting and rescaling
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
