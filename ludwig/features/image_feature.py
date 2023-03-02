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
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import normalize

from ludwig.constants import (
    CHECKSUM,
    COLUMN,
    ENCODER,
    HEIGHT,
    IMAGE,
    IMAGENET1K,
    INFER_IMAGE_DIMENSIONS,
    INFER_IMAGE_MAX_HEIGHT,
    INFER_IMAGE_MAX_WIDTH,
    INFER_IMAGE_SAMPLE_SIZE,
    NAME,
    NUM_CHANNELS,
    PREPROCESSING,
    PROC_COLUMN,
    REQUIRES_EQUAL_DIMENSIONS,
    SRC,
    TRAINING,
    TYPE,
    WIDTH,
)
from ludwig.data.cache.types import wrap
from ludwig.encoders.image.torchvision import TVModelVariant
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.image import (
    RandomBlurConfig,
    RandomBrightnessConfig,
    RandomContrastConfig,
    RandomHorizontalFlipConfig,
    RandomRotateConfig,
    RandomVerticalFlipConfig,
)
from ludwig.schema.features.image_feature import ImageInputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.augmentation_utils import get_augmentation_op, register_augmentation_op
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.dataframe_utils import is_dask_series_or_df
from ludwig.utils.fs_utils import has_remote_protocol, upload_h5
from ludwig.utils.image_utils import (
    get_gray_default_image,
    grayscale,
    is_torchvision_encoder,
    num_channels_in_image,
    read_image_from_bytes_obj,
    read_image_from_path,
    resize_image,
    ResizeChannels,
    torchvision_model_registry,
)
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.types import Series, TorchscriptPreprocessingInput

# constants used for Ludwig image preprocessing
IMAGENET1K_MEAN = [0.485, 0.456, 0.406]
IMAGENET1K_STD = [0.229, 0.224, 0.225]


logger = logging.getLogger(__name__)


###
# Image specific augmentation operations
###
@register_augmentation_op(name="random_vertical_flip", features=IMAGE)
class RandomVFlip(torch.nn.Module):
    def __init__(
        self,
        config: RandomVerticalFlipConfig,
    ):
        super().__init__()

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            imgs = F.vflip(imgs)

        return imgs


@register_augmentation_op(name="random_horizontal_flip", features=IMAGE)
class RandomHFlip(torch.nn.Module):
    def __init__(
        self,
        config: RandomHorizontalFlipConfig,
    ):
        super().__init__()

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            imgs = F.hflip(imgs)

        return imgs


@register_augmentation_op(name="random_rotate", features=IMAGE)
class RandomRotate(torch.nn.Module):
    def __init__(self, config: RandomRotateConfig):
        super().__init__()
        self.degree = config.degree

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            # map angle to interval (-degree, +degree)
            angle = (torch.rand(1) * 2 * self.degree - self.degree).item()
            return F.rotate(imgs, angle)
        else:
            return imgs


@register_augmentation_op(name="random_contrast", features=IMAGE)
class RandomContrast(torch.nn.Module):
    def __init__(self, config: RandomContrastConfig):
        super().__init__()
        self.min_contrast = config.min
        self.contrast_adjustment_range = config.max - config.min

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            # random contrast adjustment
            adjust_factor = (torch.rand(1) * self.contrast_adjustment_range + self.min_contrast).item()
            return F.adjust_contrast(imgs, adjust_factor)
        else:
            return imgs


@register_augmentation_op(name="random_brightness", features=IMAGE)
class RandomBrightness(torch.nn.Module):
    def __init__(self, config: RandomBrightnessConfig):
        super().__init__()
        self.min_brightness = config.min
        self.brightness_adjustment_range = config.max - config.min

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            # random contrast adjustment
            adjust_factor = (torch.rand(1) * self.brightness_adjustment_range + self.min_brightness).item()
            return F.adjust_brightness(imgs, adjust_factor)
        else:
            return imgs


@register_augmentation_op(name="random_blur", features=IMAGE)
class RandomBlur(torch.nn.Module):
    def __init__(self, config: RandomBlurConfig):
        super().__init__()
        self.kernel_size = [config.kernel_size, config.kernel_size]

    def forward(self, imgs):
        if torch.rand(1) < 0.5:
            imgs = F.gaussian_blur(imgs, self.kernel_size)

        return imgs


class ImageAugmentation(torch.nn.Module):
    def __init__(
        self,
        augmentation_list: List[BaseAugmentationConfig],
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()

        # TODO: change to debug level before merging
        logger.info(f"Creating Augmentation pipline: {augmentation_list}")

        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if self.training:
            self.augmentation_steps = torch.nn.Sequential()
            for aug_config in augmentation_list:
                try:
                    aug_op = get_augmentation_op(IMAGE, aug_config.type)
                    self.augmentation_steps.append(aug_op(aug_config))
                except KeyError:
                    raise ValueError(f"Invalid augmentation operation specification: {aug_config}")
        else:
            # TODO: should this raise an exception if not in training mode?
            self.augmentation_steps = None

    def forward(self, imgs):
        if self.augmentation_steps:
            # convert from float to uint8 values - this is required for the augmentation
            imgs = self._convert_back_to_uint8(imgs)

            logger.debug(f"Executing augmentation pipeline steps:\n{self.augmentation_steps}")
            imgs = self.augmentation_steps(imgs)

            # convert back to float32 values and renormalize if needed
            imgs = self._renormalize_image(imgs)

        return imgs

    # function to partially undo the TorchVision ImageClassification transformation.
    #  back out the normalization step and convert from float32 to uint8 dtype
    #  to make the tensor displayable as an image
    #  crop size remains the same
    def _convert_back_to_uint8(self, images):
        if self.normalize_mean:
            mean = torch.as_tensor(self.normalize_mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.as_tensor(self.normalize_std, dtype=torch.float32).view(-1, 1, 1)
            return images.mul(std).add(mean).mul(255.0).type(torch.uint8)
        else:
            return images.mul(255.0).type(torch.uint8)

    # function to redo part of the TorchVision ImageClassification transformation.
    #  convert uint8 to float32
    #  apply the imagenet1k normalization
    def _renormalize_image(self, images):
        if self.normalize_mean:
            mean = torch.as_tensor(self.normalize_mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.as_tensor(self.normalize_std, dtype=torch.float32).view(-1, 1, 1)
            return images.type(torch.float32).div(255.0).sub(mean).div(std)
        else:
            return images.type(torch.float32).div(255.0)


@dataclass
class ImageTransformMetadata:
    height: int
    width: int
    num_channels: int


def _get_torchvision_transform(
    torchvision_parameters: TVModelVariant,
) -> Tuple[torch.nn.Module, ImageTransformMetadata]:
    """Returns a torchvision transform that is compatible with the model variant.

    Note that the raw torchvision transform is not returned. Instead, a Sequential module that includes
    image resizing is returned. This is because the raw torchvision transform assumes that the input image has
    three channels, which is not always the case with images input into Ludwig.

    Args:
        torchvision_parameters: The parameters for the torchvision model variant.
    Returns:
        (torchvision_transform, transform_metadata): A torchvision transform and the metadata for the transform.
    """
    torchvision_transform_raw = torchvision_parameters.model_weights.DEFAULT.transforms()
    torchvision_transform = torch.nn.Sequential(
        ResizeChannels(num_channels=3),
        torchvision_transform_raw,
    )
    transform_metadata = ImageTransformMetadata(
        height=torchvision_transform_raw.crop_size[0],
        width=torchvision_transform_raw.crop_size[0],
        num_channels=len(torchvision_transform_raw.mean),
    )
    return (torchvision_transform, transform_metadata)


def _get_torchvision_parameters(model_type: str, model_variant: str) -> TVModelVariant:
    return torchvision_model_registry.get(model_type).get(model_variant)


class _ImagePreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by ImageFeatureMixin.add_feature_data."""

    def __init__(
        self,
        metadata: TrainingSetMetadataDict,
        torchvision_transform: Optional[torch.nn.Module] = None,
        transform_metadata: Optional[ImageTransformMetadata] = None,
    ):
        super().__init__()

        self.resize_method = metadata["preprocessing"]["resize_method"]
        self.torchvision_transform = torchvision_transform
        if transform_metadata is not None:
            self.height = transform_metadata.height
            self.width = transform_metadata.width
            self.num_channels = transform_metadata.num_channels
        else:
            self.height = metadata["preprocessing"]["height"]
            self.width = metadata["preprocessing"]["width"]
            self.num_channels = metadata["preprocessing"]["num_channels"]

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        """Takes a list of images and adjusts the size and number of channels as specified in the metadata.

        If `v` is already a torch.Tensor, we assume that the images are already preprocessed to be the same size.
        """
        # Nested conditional is a workaround to short-circuit boolean evaluation.
        if not torch.jit.isinstance(v, List[torch.Tensor]):
            if not torch.jit.isinstance(v, torch.Tensor):
                raise ValueError(f"Unsupported input: {v}")

        if self.torchvision_transform is not None:
            # perform pre-processing for torchvision pretrained model encoders
            if torch.jit.isinstance(v, List[torch.Tensor]):
                imgs = [self.torchvision_transform(img) for img in v]
            else:
                # convert batch of image tensors to a list and then run torchvision pretrained
                # model transforms on each image
                imgs = [self.torchvision_transform(img) for img in torch.unbind(v)]

            # collect the list of images into a batch
            imgs_stacked = torch.stack(imgs)
        else:
            # perform pre-processing for Ludwig defined image encoders
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

            imgs_stacked = imgs_stacked.type(torch.float32) / 255

        return imgs_stacked


class ImageFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return IMAGE

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        return {PREPROCESSING: preprocessing_parameters}

    @staticmethod
    def _read_image_if_bytes_obj_and_resize(
        img_entry: Union[bytes, torch.Tensor, np.ndarray],
        img_width: int,
        img_height: int,
        should_resize: bool,
        num_channels: int,
        resize_method: str,
        user_specified_num_channels: bool,
        standardize_image: str,
    ) -> Optional[np.ndarray]:
        """
        :param img_entry Union[bytes, torch.Tensor, np.ndarray]: if str file path to the
            image else torch.Tensor of the image itself
        :param img_width: expected width of the image
        :param img_height: expected height of the image
        :param should_resize: Should the image be resized?
        :param resize_method: type of resizing method
        :param num_channels: expected number of channels in the first image
        :param user_specified_num_channels: did the user specify num channels?
        :param standardize_image: specifies whether to standarize image with imagenet1k specifications
        :return: image object as a numpy array

        Helper method to read and resize an image according to model definition.
        If the user doesn't specify a number of channels, we use the first image
        in the dataset as the source of truth. If any image in the dataset
        doesn't have the same number of channels as the first image,
        raise an exception.

        If the user specifies a number of channels, we try to convert all the
        images to the specifications by dropping channels/padding 0 channels
        """

        if isinstance(img_entry, bytes):
            img = read_image_from_bytes_obj(img_entry, num_channels)
        elif isinstance(img_entry, str):
            img = read_image_from_path(img_entry, num_channels)
        elif isinstance(img_entry, np.ndarray):
            img = torch.from_numpy(np.array(img_entry, copy=True)).permute(2, 0, 1)
        else:
            img = img_entry

        if not isinstance(img, torch.Tensor):
            warnings.warn(f"Image with value {img} cannot be read")
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

        # casting and rescaling
        img = img.type(torch.float32) / 255

        if standardize_image == IMAGENET1K:
            img = normalize(img, mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD)

        return img.numpy()

    @staticmethod
    def _read_image_with_pretrained_transform(
        img_entry: Union[bytes, torch.Tensor, np.ndarray],
        transform_fn: Callable,
    ) -> Optional[np.ndarray]:

        if isinstance(img_entry, bytes):
            img = read_image_from_bytes_obj(img_entry)
        elif isinstance(img_entry, str):
            img = read_image_from_path(img_entry)
        elif isinstance(img_entry, np.ndarray):
            img = torch.from_numpy(img_entry).permute(2, 0, 1)
        else:
            img = img_entry

        if not isinstance(img, torch.Tensor):
            warnings.warn(f"Image with value {img} cannot be read")
            return None

        img = transform_fn(img)

        return img.numpy()

    @staticmethod
    def _set_image_and_height_equal_for_encoder(
        width: int, height: int, preprocessing_parameters: dict, encoder_type: str
    ) -> Tuple[int, int]:
        """Some pretrained image encoders require images with the same dimension, or images with a specific width
        and heigh values. The returned width and height are set based on compatibility with the downstream encoder
        using the encoder parameters for the feature.

        Args:
            width: Represents the width of the image. This is either specified in the user config, or inferred using
                a sample of images.
            height: Represents the height of the image. This is either specified in the user config, or inferred using
                a sample of images.
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder

        Return:
            (width, height) Updated width and height so that they are equal
        """

        if preprocessing_parameters[REQUIRES_EQUAL_DIMENSIONS] and height != width:
            width = height = min(width, height)
            # Update preprocessing parameters dictionary to reflect new height and width values
            preprocessing_parameters["width"] = width
            preprocessing_parameters["height"] = height
            logger.info(
                f"Set image feature height and width to {width} to be compatible with" f" {encoder_type} encoder."
            )
        return width, height

    @staticmethod
    def _infer_image_size(
        image_sample: List[torch.Tensor],
        max_height: int,
        max_width: int,
        preprocessing_parameters: dict,
        encoder_type: str,
    ) -> Tuple[int, int]:
        """Infers the size to use from a group of images. The returned height will be the average height of images
        in image_sample rounded to the nearest integer, or max_height. Likewise for width.

        Args:
            image_sample: Sample of images to use to infer image size. Must be formatted as [channels, height, width].
            max_height: Maximum height.
            max_width: Maximum width.
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder

        Return:
            (height, width) The inferred height and width.
        """

        height_avg = sum(x.shape[1] for x in image_sample) / len(image_sample)
        width_avg = sum(x.shape[2] for x in image_sample) / len(image_sample)
        height = min(int(round(height_avg)), max_height)
        width = min(int(round(width_avg)), max_width)

        # Update height and width if the downstream encoder requires images
        # with  the same dimension or specific width and height values
        width, height = ImageFeatureMixin._set_image_and_height_equal_for_encoder(
            width, height, preprocessing_parameters, encoder_type
        )

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
        logger.info(f"Inferring num_channels from the first {n_images} images.")
        logger.info("\n".join([f"  images with {k} channels: {v}" for k, v in sorted(channel_frequency.items())]))
        if num_channels == max(channel_frequency, key=channel_frequency.get):
            logger.info(
                f"Using {num_channels} channels because it is the majority in sample. If an image with"
                f" a different depth is read, will attempt to convert to {num_channels} channels."
            )
        else:
            logger.info(f"Defaulting to {num_channels} channels.")
        logger.info(
            "To explicitly set the number of channels, define num_channels in the preprocessing dictionary of "
            "the image input feature config."
        )
        return num_channels

    @staticmethod
    def _finalize_preprocessing_parameters(
        preprocessing_parameters: dict,
        encoder_type: str,
        column: Series,
    ) -> Tuple:
        """Helper method to determine the height, width and number of channels for preprocessing the image data.

        This is achieved by looking at the parameters provided by the user. When there are some missing parameters, we
        fall back on to the first image in the dataset. The assumption being that all the images in the data are
        expected be of the same size with the same number of channels.

        Args:
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder
            column: The data itself. Can be a Pandas, Modin or Dask series.
        """

        explicit_height_width = preprocessing_parameters[HEIGHT] or preprocessing_parameters[WIDTH]
        explicit_num_channels = NUM_CHANNELS in preprocessing_parameters and preprocessing_parameters[NUM_CHANNELS]

        if preprocessing_parameters[INFER_IMAGE_DIMENSIONS] and not (explicit_height_width and explicit_num_channels):
            sample_size = min(len(column), preprocessing_parameters[INFER_IMAGE_SAMPLE_SIZE])
        else:
            sample_size = 1  # Take first image

        sample = []
        sample_num_bytes = []
        failed_entries = []
        for image_entry in column.head(sample_size):
            if isinstance(image_entry, str):
                # Tries to read image as PNG or numpy file from the path.
                image, num_bytes = read_image_from_path(image_entry, return_num_bytes=True)
                if num_bytes is not None:
                    sample_num_bytes.append(num_bytes)
            else:
                image = image_entry

            if isinstance(image, torch.Tensor):
                sample.append(image)
            elif isinstance(image, np.ndarray):
                sample.append(torch.from_numpy(image).permute(2, 0, 1))
            else:
                failed_entries.append(image_entry)
        if len(sample) == 0:
            failed_entries_repr = "\n\t- ".join(failed_entries)
            raise ValueError(
                f"Images dimensions cannot be inferred. Failed to read {sample_size} images as samples:\n\t- "
                f"{failed_entries_repr}."
            )

        should_resize = False
        if explicit_height_width:
            should_resize = True
            try:
                height = int(preprocessing_parameters[HEIGHT])
                width = int(preprocessing_parameters[WIDTH])
                # Update height and width if the downstream encoder requires images
                # with the same dimension or specific width and height values
                width, height = ImageFeatureMixin._set_image_and_height_equal_for_encoder(
                    width, height, preprocessing_parameters, encoder_type
                )
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
                    sample,
                    max_height=preprocessing_parameters[INFER_IMAGE_MAX_HEIGHT],
                    max_width=preprocessing_parameters[INFER_IMAGE_MAX_WIDTH],
                    preprocessing_parameters=preprocessing_parameters,
                    encoder_type=encoder_type,
                )
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
                num_channels = ImageFeatureMixin._infer_number_of_channels(sample)
            elif len(sample) > 0:
                num_channels = num_channels_in_image(sample[0])
            else:
                raise ValueError(
                    "Explicit image num channels is not set, infer_image_dimensions is false, "
                    "and first image cannot be read, so image num channels is unknown"
                )

        assert isinstance(num_channels, int), ValueError("Number of image channels needs to be an integer")

        average_file_size = np.mean(sample_num_bytes) if sample_num_bytes else None

        standardize_image = preprocessing_parameters["standardize_image"]
        if standardize_image == "imagenet1k" and num_channels != 3:
            warnings.warn(
                f"'standardize_image=imagenet1k' is defined only for 'num_channels=3' but "
                f"detected 'num_channels={num_channels}'.  For this situation setting 'standardize_image=None'.",
                RuntimeWarning,
            )
            standardize_image = None

        return (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            average_file_size,
            standardize_image,
        )

    @staticmethod
    def add_feature_data(
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        set_default_value(feature_config[PREPROCESSING], "in_memory", preprocessing_parameters["in_memory"])

        name = feature_config[NAME]
        column = input_df[feature_config[COLUMN]]
        encoder_type = feature_config[ENCODER][TYPE]

        src_path = None
        if SRC in metadata:
            src_path = os.path.dirname(os.path.abspath(metadata.get(SRC)))
        abs_path_column = backend.df_engine.map_objects(
            column,
            lambda row: get_abs_path(src_path, row) if isinstance(row, str) and not has_remote_protocol(row) else row,
        )

        # determine if specified encoder is a torchvision model
        model_type = feature_config[ENCODER].get("type", None)
        model_variant = feature_config[ENCODER].get("model_variant")
        if model_variant:
            torchvision_parameters = _get_torchvision_parameters(model_type, model_variant)
        else:
            torchvision_parameters = None

        if torchvision_parameters:
            logger.warning(
                f"Using the transforms specified for the torchvision model {model_type} {model_variant} "
                f"This includes setting the number of channels is 3 and resizing the image to the needs of the model."
            )

            torchvision_transform, transform_metadata = _get_torchvision_transform(torchvision_parameters)

            # torchvision_parameters is not None
            # perform torchvision model transformations
            read_image_if_bytes_obj_and_resize = partial(
                ImageFeatureMixin._read_image_with_pretrained_transform,
                transform_fn=torchvision_transform,
            )
            average_file_size = None

            # save weight specification in preprocessing section
            preprocessing_parameters[
                "torchvision_model_default_weights"
            ] = f"{torchvision_parameters.model_weights.DEFAULT}"

            # add torchvision model id to preprocessing section for torchscript
            preprocessing_parameters["torchvision_model_type"] = model_type
            preprocessing_parameters["torchvision_model_variant"] = model_variant

            # get required setup parameters for in_memory = False processing
            height = transform_metadata.height
            width = transform_metadata.width
            num_channels = transform_metadata.num_channels
        else:
            # torchvision_parameters is None
            # perform Ludwig specified transformations
            (
                should_resize,
                width,
                height,
                num_channels,
                user_specified_num_channels,
                average_file_size,
                standardize_image,
            ) = ImageFeatureMixin._finalize_preprocessing_parameters(
                preprocessing_parameters, encoder_type, abs_path_column
            )

            metadata[name][PREPROCESSING]["height"] = height
            metadata[name][PREPROCESSING]["width"] = width
            metadata[name][PREPROCESSING]["num_channels"] = num_channels

            read_image_if_bytes_obj_and_resize = partial(
                ImageFeatureMixin._read_image_if_bytes_obj_and_resize,
                img_width=width,
                img_height=height,
                should_resize=should_resize,
                num_channels=num_channels,
                resize_method=preprocessing_parameters["resize_method"],
                user_specified_num_channels=user_specified_num_channels,
                standardize_image=standardize_image,
            )

        # TODO: alternatively use get_average_image() for unreachable images
        default_image = get_gray_default_image(num_channels, height, width)
        metadata[name]["reshape"] = (num_channels, height, width)

        in_memory = feature_config[PREPROCESSING]["in_memory"]
        if in_memory or skip_save_processed_input:

            proc_col = backend.read_binary_files(
                abs_path_column, map_fn=read_image_if_bytes_obj_and_resize, file_size=average_file_size
            )

            num_failed_image_reads = (
                proc_col.isna().sum().compute() if is_dask_series_or_df(proc_col, backend) else proc_col.isna().sum()
            )

            proc_col = backend.df_engine.map_objects(
                proc_col, lambda row: default_image if not isinstance(row, np.ndarray) else row
            )

            proc_df[feature_config[PROC_COLUMN]] = proc_col
        else:
            num_images = len(abs_path_column)
            num_failed_image_reads = 0

            data_fp = backend.cache.get_cache_path(wrap(metadata.get(SRC)), metadata.get(CHECKSUM), TRAINING)
            with upload_h5(data_fp) as h5_file:
                # todo future add multiprocessing/multithreading
                image_dataset = h5_file.create_dataset(
                    feature_config[PROC_COLUMN] + "_data", (num_images, num_channels, height, width), dtype=np.float32
                )
                for i, img_entry in enumerate(abs_path_column):
                    res = read_image_if_bytes_obj_and_resize(img_entry)
                    if isinstance(res, np.ndarray):
                        image_dataset[i, :height, :width, :] = res
                    else:
                        logger.warning(f"Failed to read image {img_entry} while preprocessing feature `{name}`. ")
                        image_dataset[i, :height, :width, :] = default_image
                        num_failed_image_reads += 1
                h5_file.flush()

            proc_df[feature_config[PROC_COLUMN]] = np.arange(num_images)

        if num_failed_image_reads > 0:
            logger.warning(
                f"Failed to read {num_failed_image_reads} images while preprocessing feature `{name}`. "
                "Using default image for these rows in the dataset."
            )

        return proc_df


class ImageInputFeature(ImageFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: ImageInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

        # set up for augmentation if it is enabled
        if input_feature_config.augmentation:
            # assume no image normalize is required
            normalize_mean = normalize_std = None

            # determine if specified encoder is a torchvision model
            if is_torchvision_encoder(self.encoder_obj):
                # encoder is a torchvision model
                normalize_mean = self.encoder_obj.normalize_mean
                normalize_std = self.encoder_obj.normalize_std
            else:
                # encoder is a Ludwig encoder, determine if standardize_image is set to IMAGENET1K
                if input_feature_config.preprocessing.standardize_image == IMAGENET1K:
                    normalize_mean = IMAGENET1K_MEAN
                    normalize_std = IMAGENET1K_STD

            # create augmentation pipeline object
            self.augmentation_pipeline = ImageAugmentation(
                input_feature_config.augmentation,
                normalize_mean,
                normalize_std,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(inputs, torch.Tensor), f"inputs to image feature must be a torch tensor, got {type(inputs)}"
        assert inputs.dtype in [torch.float32], f"inputs to image feature must be a float32 tensor, got {inputs.dtype}"

        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    @property
    def input_dtype(self):
        return torch.float32

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self.encoder_obj.input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    def update_config_after_module_init(self, feature_config):
        if is_torchvision_encoder(self.encoder_obj):
            # update feature preprocessing parameters to reflect used in torchvision pretrained model
            # Note: image height and width is determined by the encoder crop_size attribute.  Source of this
            # attribute is from the torchvision.transforms._presets.ImageClassification class.  This class stores
            # crop_size as a single element list.  the single element in this list is used to set both the height
            # and width of an image.
            feature_config.preprocessing.height = self.encoder_obj.crop_size[0]
            feature_config.preprocessing.width = self.encoder_obj.crop_size[0]
            feature_config.preprocessing.num_channels = self.encoder_obj.num_channels

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        for key in ["height", "width", "num_channels", "standardize_image"]:
            if hasattr(feature_config.encoder, key):
                setattr(feature_config.encoder, key, feature_metadata[PREPROCESSING][key])

    @staticmethod
    def get_schema_cls():
        return ImageInputFeatureConfig

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        model_type = metadata["preprocessing"].get("torchvision_model_type")
        model_variant = metadata["preprocessing"].get("torchvision_model_variant")
        if model_variant:
            torchvision_parameters = _get_torchvision_parameters(model_type, model_variant)
        else:
            torchvision_parameters = None

        if torchvision_parameters:
            torchvision_transform, transform_metadata = _get_torchvision_transform(torchvision_parameters)
        else:
            torchvision_transform = None
            transform_metadata = None

        return _ImagePreprocessing(
            metadata, torchvision_transform=torchvision_transform, transform_metadata=transform_metadata
        )

    def get_augmentation_pipeline(self):
        return self.augmentation_pipeline
