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
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torchvision

from ludwig.constants import (
    BACKFILL,
    CHECKSUM,
    COLUMN,
    HEIGHT,
    IMAGE,
    INFER_IMAGE_DIMENSIONS,
    INFER_IMAGE_MAX_HEIGHT,
    INFER_IMAGE_MAX_WIDTH,
    INFER_IMAGE_SAMPLE_SIZE,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NAME,
    NUM_CHANNELS,
    PREPROCESSING,
    PROC_COLUMN,
    RESIZE_METHODS,
    SRC,
    TIED,
    TRAINING,
    WIDTH,
)
from ludwig.data.cache.types import wrap
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import has_remote_protocol, upload_h5
from ludwig.utils.image_utils import (
    get_gray_default_image,
    get_image_from_path,
    grayscale,
    num_channels_in_image,
    read_image,
    resize_image,
)
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)


# TODO(shreya): Confirm if it's ok to do per channel normalization
# TODO(shreya): Also confirm if this is being used anywhere
# TODO(shreya): Confirm if ok to use imagenet means and std devs
image_scaling_registry = {
    "pixel_normalization": lambda x: x * 1.0 / 255,
    "pixel_standardization": partial(
        torchvision.transforms.functional.normalize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
}


class _ImagePreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by ImageFeatureMixin.add_feature_data."""

    def __init__(self, metadata: Dict[str, Any]):
        super().__init__()
        self.height = metadata["preprocessing"]["height"]
        self.width = metadata["preprocessing"]["width"]
        self.num_channels = metadata["preprocessing"]["num_channels"]
        self.resize_method = metadata["preprocessing"]["resize_method"]

    def forward(self, v: Union[List[str], List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Takes a list of images and adjusts the size and number of channels as specified in the metadata.

        If `v` is already a torch.Tensor, we assume that the images are already preprocessed to be the same size.
        """
        if torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        if torch.jit.isinstance(v, List[torch.Tensor]):
            imgs = [resize_image(img, (self.height, self.width), self.resize_method) for img in v]
            imgs_stacked = torch.stack(imgs)
        else:
            imgs_stacked = v

        _, num_channels, height, width = imgs_stacked.shape

        # Ensure images are the size expected by the model
        if height != self.height or width != self.width:
            imgs_stacked = resize_image(imgs_stacked, (self.height, self.width), self.resize_method)

        # Ensures images have the number of channels expected by the model
        if num_channels != self.num_channels:
            if self.num_channels == 1:
                imgs_stacked = grayscale(imgs_stacked)
            elif num_channels < self.num_channels:
                extra_channels = self.num_channels - num_channels
                imgs_stacked = torch.nn.functional.pad(imgs_stacked, [0, 0, 0, 0, 0, extra_channels])
            else:
                raise ValueError(
                    f"Number of channels cannot be reconciled. metadata.num_channels = "
                    f"{self.num_channels}, but imgs.shape[1] = {num_channels}"
                )

        return imgs_stacked


class ImageFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return IMAGE

    @staticmethod
    def preprocessing_defaults():
        return {
            "missing_value_strategy": BACKFILL,
            "in_memory": True,
            "resize_method": "interpolate",
            "scaling": "pixel_normalization",
            "num_processes": 1,
            "infer_image_num_channels": True,
            "infer_image_dimensions": True,
            "infer_image_max_height": 256,
            "infer_image_max_width": 256,
            "infer_image_sample_size": 100,
        }

    @staticmethod
    def preprocessing_schema():
        return {
            "missing_value_strategy": {"type": "string", "enum": MISSING_VALUE_STRATEGY_OPTIONS},
            "in_memory": {"type": "boolean"},
            "resize_method": {"type": "string", "enum": RESIZE_METHODS},
            "scaling": {"type": "string", "enum": list(image_scaling_registry.keys())},
            "num_processes": {"type": "integer", "minimum": 0},
            "height": {"type": "integer", "minimum": 0},
            "width": {"type": "integer", "minimum": 0},
            "num_channels": {"type": "integer", "minimum": 0},
            "infer_image_num_channels": {"type": "boolean"},
            "infer_image_dimensions": {"type": "boolean"},
            "infer_image_max_height": {"type": "integer", "minimum": 0},
            "infer_image_max_width": {"type": "integer", "minimum": 0},
            "infer_image_sample_size": {"type": "integer", "minimum": 0},
        }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {PREPROCESSING: preprocessing_parameters}

    @staticmethod
    def _read_image_and_resize(
        img_entry: Union[str, torch.Tensor],
        img_width: int,
        img_height: int,
        should_resize: bool,
        num_channels: int,
        resize_method: str,
        user_specified_num_channels: bool,
    ) -> Optional[np.ndarray]:
        """
        :param img_entry Union[str, 'numpy.array']: if str file path to the
                image else numpy.array of the image itself
        :param img_width: expected width of the image
        :param img_height: expected height of the image
        :param should_resize: Should the image be resized?
        :param resize_method: type of resizing method
        :param num_channels: expected number of channels in the first image
        :param user_specified_num_channels: did the user specify num channels?
        :return: image object as a numpy array

        Helper method to read and resize an image according to model definition.
        If the user doesn't specify a number of channels, we use the first image
        in the dataset as the source of truth. If any image in the dataset
        doesn't have the same number of channels as the first image,
        raise an exception.

        If the user specifies a number of channels, we try to convert all the
        images to the specifications by dropping channels/padding 0 channels
        """

        img = read_image(img_entry, num_channels)
        if img is None:
            logger.info(f"{img_entry} cannot be read")
            return None
        img_num_channels = num_channels_in_image(img)
        # Convert to grayscale if needed.
        if num_channels == 1 and img_num_channels != 1:
            img = grayscale(img)
            img_num_channels = 1

        if should_resize:
            img = resize_image(img, (img_height, img_width), resize_method)

        if user_specified_num_channels:
            # Number of channels is specified by the user
            # img_padded = np.zeros((img_height, img_width, num_channels),
            #                       dtype=np.uint8)
            # min_num_channels = min(num_channels, img_num_channels)
            # img_padded[:, :, :min_num_channels] = img[:, :, :min_num_channels]
            # img = img_padded
            if num_channels > img_num_channels:
                extra_channels = num_channels - img_num_channels
                img = torch.nn.functional.pad(img, [0, 0, 0, 0, 0, extra_channels])

            if img_num_channels != num_channels:
                logger.warning(
                    "Image has {} channels, where as {} "
                    "channels are expected. Dropping/adding channels "
                    "with 0s as appropriate".format(img_num_channels, num_channels)
                )
        else:
            # If the image isn't like the first image, raise exception
            if img_num_channels != num_channels:
                raise ValueError(
                    "Image has {} channels, unlike the first image, which "
                    "has {} channels. Make sure all the images have the same "
                    "number of channels or use the num_channels property in "
                    "image preprocessing".format(img_num_channels, num_channels)
                )

        if img.shape[1] != img_height or img.shape[2] != img_width:
            raise ValueError(
                "Images are not of the same size. "
                "Expected size is {}, "
                "current image size is {}."
                "Images are expected to be all of the same size "
                "or explicit image width and height are expected "
                "to be provided. "
                "Additional information: "
                "https://ludwig-ai.github.io/ludwig-docs/latest/configuration/features/image_features"
                "#image-features-preprocessing".format([img_height, img_width, num_channels], img.shape)
            )

        return img.numpy()

    @staticmethod
    def _infer_image_size(image_sample: List[torch.Tensor], max_height: int, max_width: int):
        """Infers the size to use from a group of images. The returned height will be the average height of images
        in image_sample rounded to the nearest integer, or max_height. Likewise for width.

        Args:
            image_sample: Sample of images to use to infer image size. Must be formatted as [channels, height, width].
            max_height: Maximum height.
            max_width: Maximum width.

        Return:
            (height, width) The inferred height and width.
        """
        height_avg = sum(x.shape[1] for x in image_sample) / len(image_sample)
        width_avg = sum(x.shape[2] for x in image_sample) / len(image_sample)
        height = min(int(round(height_avg)), max_height)
        width = min(int(round(width_avg)), max_width)

        logger.debug(f"Inferring height: {height} and width: {width}")
        return height, width

    @staticmethod
    def _infer_number_of_channels(image_sample: List[torch.Tensor]):
        """Infers the channel depth to use from a group of images.

        We make the assumption that the majority of datasets scraped from the web will be RGB, so if we get a mixed bag
        of images we should default to that. However, if the majority of the sample images have a specific channel depth
        (other than 3) this is probably intentional so we keep it, but log an info message.
        """
        n_images = len(image_sample)
        channel_frequency = Counter([num_channels_in_image(x) for x in image_sample])
        if channel_frequency[1] > n_images / 2:
            # If the majority of images in sample are 1 channel, use 1.
            num_channels = 1
        elif channel_frequency[2] > n_images / 2:
            # If the majority of images in sample are 2 channel, use 2.
            num_channels = 2
        elif channel_frequency[4] > n_images / 2:
            # If the majority of images in sample are 4 channel, use 4.
            num_channels = 4
        else:
            # Default case: use 3 channels.
            num_channels = 3
        logging.info(f"Inferring num_channels from the first {n_images} images.")
        logging.info("\n".join([f"  images with {k} channels: {v}" for k, v in sorted(channel_frequency.items())]))
        if num_channels == max(channel_frequency, key=channel_frequency.get):
            logging.info(
                f"Using {num_channels} channels because it is the majority in sample. If an image with"
                f" a different depth is read, will attempt to convert to {num_channels} channels."
            )
        else:
            logging.info(f"Defaulting to {num_channels} channels.")
        logging.info(
            "To explicitly set the number of channels, define num_channels in the preprocessing dictionary of "
            "the image input feature config."
        )
        return num_channels

    @staticmethod
    def _finalize_preprocessing_parameters(
        preprocessing_parameters: dict,
        first_img_entry: Optional[Union[str, torch.Tensor]],
        src_path: str,
        input_feature_col: np.array,
    ) -> Tuple:
        """Helper method to determine the height, width and number of channels for preprocessing the image data.

        This is achieved by looking at the parameters provided by the user. When there are some missing parameters, we
        fall back on to the first image in the dataset. The assumption being that all the images in the data are
        expected be of the same size with the same number of channels
        """

        explicit_height_width = (
            HEIGHT in preprocessing_parameters or WIDTH in preprocessing_parameters
        ) and first_img_entry is not None
        explicit_num_channels = (
            NUM_CHANNELS in preprocessing_parameters and preprocessing_parameters[NUM_CHANNELS]
        ) and first_img_entry is not None

        if explicit_num_channels:
            first_image = read_image(first_img_entry, preprocessing_parameters[NUM_CHANNELS])
        else:
            first_image = read_image(first_img_entry)

        inferred_sample = None
        if preprocessing_parameters[INFER_IMAGE_DIMENSIONS] and not (explicit_height_width and explicit_num_channels):
            sample_size = min(len(input_feature_col), preprocessing_parameters[INFER_IMAGE_SAMPLE_SIZE])
            sample = []
            for img in input_feature_col.head(sample_size):
                try:
                    sample.append(read_image(get_image_from_path(src_path, img, ret_bytes=True)))
                except requests.exceptions.HTTPError:
                    pass
            inferred_sample = [img for img in sample if img is not None]
            if not inferred_sample:
                raise ValueError("No readable images in sample, image dimensions cannot be inferred")

        should_resize = False
        if explicit_height_width:
            should_resize = True
            try:
                height = int(preprocessing_parameters[HEIGHT])
                width = int(preprocessing_parameters[WIDTH])
            except ValueError as e:
                raise ValueError("Image height and width must be set and have " "positive integer values: " + str(e))
            if height <= 0 or width <= 0:
                raise ValueError("Image height and width must be positive integers")
        else:
            # User hasn't specified height and width.
            # Default to inferring from sample or first image.
            if preprocessing_parameters[INFER_IMAGE_DIMENSIONS]:
                should_resize = True
                height, width = ImageFeatureMixin._infer_image_size(
                    inferred_sample,
                    max_height=preprocessing_parameters[INFER_IMAGE_MAX_HEIGHT],
                    max_width=preprocessing_parameters[INFER_IMAGE_MAX_WIDTH],
                )
            elif first_image is not None:
                height, width = first_image.shape[0], first_image.shape[1]
            else:
                raise ValueError(
                    "Explicit image width/height are not set, infer_image_dimensions is false, "
                    "and first image cannot be read, so image dimensions are unknown"
                )

        if explicit_num_channels:
            # User specified num_channels in the model/feature config
            user_specified_num_channels = True
            num_channels = preprocessing_parameters[NUM_CHANNELS]
        else:
            user_specified_num_channels = False
            if preprocessing_parameters[INFER_IMAGE_DIMENSIONS]:
                user_specified_num_channels = True
                num_channels = ImageFeatureMixin._infer_number_of_channels(inferred_sample)
            elif first_image is not None:
                num_channels = num_channels_in_image(first_image)
            else:
                raise ValueError(
                    "Explicit image num channels is not set, infer_image_dimensions is false, "
                    "and first image cannot be read, so image num channels is unknown"
                )

        assert isinstance(num_channels, int), ValueError("Number of image channels needs to be an integer")

        return (should_resize, width, height, num_channels, user_specified_num_channels, first_image)

    @staticmethod
    def add_feature_data(
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        set_default_value(feature_config["preprocessing"], "in_memory", preprocessing_parameters["in_memory"])

        in_memory = preprocessing_parameters["in_memory"]
        if PREPROCESSING in feature_config and "in_memory" in feature_config[PREPROCESSING]:
            in_memory = feature_config[PREPROCESSING]["in_memory"]

        num_processes = preprocessing_parameters["num_processes"]
        if PREPROCESSING in feature_config and "num_processes" in feature_config[PREPROCESSING]:
            num_processes = feature_config[PREPROCESSING]["num_processes"]

        num_images = len(input_df[feature_config[COLUMN]])
        if num_images == 0:
            raise ValueError("There are no images in the dataset provided.")

        first_img_entry = next(iter(input_df[feature_config[COLUMN]]))
        logger.debug(f"Detected image feature type is {type(first_img_entry)}")

        if not isinstance(first_img_entry, str) and not isinstance(first_img_entry, torch.Tensor):
            raise ValueError(
                "Invalid image feature data type.  Detected type is {}, "
                "expect either string for file path or numpy array.".format(type(first_img_entry))
            )

        src_path = None
        if SRC in metadata:
            if isinstance(first_img_entry, str) and not has_remote_protocol(first_img_entry):
                src_path = os.path.dirname(os.path.abspath(metadata.get(SRC)))

        try:
            first_img_entry = get_image_from_path(src_path, first_img_entry, ret_bytes=True)
        except requests.exceptions.HTTPError:
            first_img_entry = None

        (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            first_image,
        ) = ImageFeatureMixin._finalize_preprocessing_parameters(
            preprocessing_parameters, first_img_entry, src_path, input_df[feature_config[COLUMN]]
        )

        metadata[feature_config[NAME]][PREPROCESSING]["height"] = height
        metadata[feature_config[NAME]][PREPROCESSING]["width"] = width
        metadata[feature_config[NAME]][PREPROCESSING]["num_channels"] = num_channels

        read_image_and_resize = partial(
            ImageFeatureMixin._read_image_and_resize,
            img_width=width,
            img_height=height,
            should_resize=should_resize,
            num_channels=num_channels,
            resize_method=preprocessing_parameters["resize_method"],
            user_specified_num_channels=user_specified_num_channels,
        )

        # TODO: alternatively use get_average_image() for unreachable images
        default_image = get_gray_default_image(num_channels, height, width)

        # check to see if the active backend can support lazy loading of
        # image features from the hdf5 cache.
        backend.check_lazy_load_supported(feature_config)

        if in_memory or skip_save_processed_input:
            # Number of processes to run in parallel for preprocessing
            metadata[feature_config[NAME]][PREPROCESSING]["num_processes"] = num_processes
            metadata[feature_config[NAME]]["reshape"] = (num_channels, height, width)

            # Split the dataset into pools only if we have an explicit request to use
            # multiple processes. In case we have multiple input images use the
            # standard code anyway.
            if backend.supports_multiprocessing and (num_processes > 1 or num_images > 1):
                all_img_entries = [
                    get_abs_path(src_path, img_entry) if isinstance(img_entry, str) else img_entry
                    for img_entry in input_df[feature_config[COLUMN]]
                ]

                with Pool(num_processes) as pool:
                    logger.debug(f"Using {num_processes} processes for preprocessing images")
                    res = pool.map(read_image_and_resize, all_img_entries)
                    proc_df[feature_config[PROC_COLUMN]] = [x if x is not None else default_image for x in res]
            else:
                # If we're not running multiple processes and we are only processing one
                # image just use this faster shortcut, bypassing multiprocessing.Pool.map
                logger.debug("No process pool initialized. Using internal process for preprocessing images")

                # helper function for handling single image
                def _get_processed_image(img_store):
                    if isinstance(img_store, str):
                        res_single = read_image_and_resize(get_abs_path(src_path, img_store))
                    else:
                        res_single = read_image_and_resize(img_store)
                    return res_single if res_single is not None else default_image

                proc_df[feature_config[PROC_COLUMN]] = backend.df_engine.map_objects(
                    input_df[feature_config[COLUMN]], _get_processed_image
                )
        else:

            all_img_entries = [
                get_abs_path(src_path, img_entry) if isinstance(img_entry, str) else img_entry
                for img_entry in input_df[feature_config[COLUMN]]
            ]

            data_fp = backend.cache.get_cache_path(wrap(metadata.get(SRC)), metadata.get(CHECKSUM), TRAINING)
            with upload_h5(data_fp) as h5_file:
                # todo future add multiprocessing/multithreading
                image_dataset = h5_file.create_dataset(
                    feature_config[PROC_COLUMN] + "_data", (num_images, num_channels, height, width), dtype=np.uint8
                )
                for i, img_entry in enumerate(all_img_entries):
                    res = read_image_and_resize(img_entry)
                    image_dataset[i, :height, :width, :] = res if res is not None else default_image
                h5_file.flush()

            proc_df[feature_config[PROC_COLUMN]] = np.arange(num_images)
        return proc_df


class ImageInputFeature(ImageFeatureMixin, InputFeature):
    height = 0
    width = 0
    num_channels = 0
    scaling = "pixel_normalization"
    encoder = "stacked_cnn"

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.uint8, torch.int64]

        # casting and rescaling
        inputs = inputs.type(torch.float32) / 255

        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    @property
    def input_dtype(self):
        return torch.uint8

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.num_channels, self.height, self.width])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        for key in ["height", "width", "num_channels", "scaling"]:
            input_feature[key] = feature_metadata[PREPROCESSING][key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, PREPROCESSING, {})

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _ImagePreprocessing(metadata)
