from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.constants import (BFILL, CROP_OR_PAD, ENCODER, ENCODER_OUTPUT,
                              ENCODER_OUTPUT_STATE, INTERPOLATE, LOGITS, TYPE)
from ludwig.features.image_feature import (ImageInputFeature,
                                           ImageOutputFeature,
                                           _ImagePreprocessing)
from ludwig.schema.features.image_feature import (ImageInputFeatureConfig,
                                                  ImageOutputFeatureConfig)
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.utils import image_feature

BATCH_SIZE = 2
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def image_config():
    return {
        "name": "image_column_name",
        "type": "image",
        "tied": None,
        "encoder": {
            "type": "stacked_cnn",
            "conv_layers": None,
            "num_conv_layers": None,
            "filter_size": 3,
            "num_filters": 256,
            "strides": (1, 1),
            "padding": "valid",
            "dilation_rate": (1, 1),
            "conv_use_bias": True,
            "conv_weights_initializer": "xavier_uniform",
            "conv_bias_initializer": "zeros",
            "conv_norm": None,
            "conv_norm_params": None,
            "conv_activation": "relu",
            "conv_dropout": 0,
            "pool_function": "max",
            "pool_size": (2, 2),
            "pool_strides": None,
            "fc_layers": None,
            "num_fc_layers": 1,
            "output_size": 16,
            "fc_use_bias": True,
            "fc_weights_initializer": "xavier_uniform",
            "fc_bias_initializer": "zeros",
            "fc_norm": None,
            "fc_norm_params": None,
            "fc_activation": "relu",
            "fc_dropout": 0,
        },
        "preprocessing": {
            "height": 28,
            "width": 28,
            "num_channels": 1,
            "scaling": "pixel_normalization",
        },
    }


@pytest.mark.parametrize(
    "encoder, height, width, num_channels",
    [
        ("stacked_cnn", 28, 28, 3),
        ("stacked_cnn", 28, 28, 1),
        ("mlp_mixer", 32, 32, 3),
    ],
)
def test_image_input_feature(image_config: Dict, encoder: str, height: int, width: int, num_channels: int) -> None:
    # setup image input feature definition
    image_def = deepcopy(image_config)
    image_def[ENCODER][TYPE] = encoder
    image_def[ENCODER]["height"] = height
    image_def[ENCODER]["width"] = width
    image_def[ENCODER]["num_channels"] = num_channels

    # pickup any other missing parameters
    defaults = ImageInputFeatureConfig(name="foo").to_dict()
    set_def = merge_dict(defaults, image_def)

    # ensure no exceptions raised during build
    image_config, _ = load_config_with_kwargs(ImageInputFeatureConfig, set_def)
    input_feature_obj = ImageInputFeature(image_config).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = torch.rand(size=(BATCH_SIZE, num_channels, height, width), dtype=torch.float32).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, *input_feature_obj.output_shape)

    # todo: remove code
    # # test for parameter updates
    # before = [(x[0], x[1].clone()) for x in input_feature_obj.named_parameters()]
    # loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(input_feature_obj.parameters(), lr=0.1)
    # target_tensor = torch.ones(encoder_output['encoder_output'].shape, dtype=torch.float32)
    #
    # # do parameter update
    # loss = loss_function(encoder_output['encoder_output'], target_tensor)
    # loss.backward()
    # optimizer.step()
    #
    # after = [(x[0], x[1].clone()) for x in input_feature_obj.named_parameters()]
    #
    # # check for parameter update
    # for b, a in zip(before, after):
    #     if not (b[1] != a[1]).any():
    #         raise RuntimeError(
    #             f'no parameter update for {a[0]}'
    #         )


@pytest.mark.parametrize(
    "encoder, decoder, height, width, num_channels, num_classes",
    [
        ("unet", "unet", 128, 128, 3, 2),
        ("unet", "unet", 32, 32, 3, 7),
    ],
)
def test_image_output_feature(
    encoder: str,
    decoder: str,
    height: int,
    width: int,
    num_channels: int,
    num_classes: int,
) -> None:
    # setup image input feature definition
    input_feature_def = image_feature(
        folder=".",
        encoder={
            "type": encoder,
            "height": height,
            "width": width,
            "num_channels": num_channels,
        },
    )
    # create image input feature object
    feature_cls = ImageInputFeature
    schema_cls = ImageInputFeatureConfig
    input_config = schema_cls.from_dict(input_feature_def)
    input_feature_obj = feature_cls(input_config).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = torch.rand(size=(BATCH_SIZE, num_channels, height, width), dtype=torch.float32).to(DEVICE)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, *input_feature_obj.output_shape)
    if encoder == "unet":
        assert len(encoder_output[ENCODER_OUTPUT_STATE]) == 4

    hidden = torch.reshape(encoder_output[ENCODER_OUTPUT], [BATCH_SIZE, -1])

    # setup image output feature definition
    output_feature_def = image_feature(
        folder=".",
        decoder={
            "type": decoder,
            "height": height,
            "width": width,
            "num_channels": num_channels,
            "num_classes": num_classes,
        },
        input_size=hidden.size(dim=1),
    )
    # create image output feature object
    feature_cls = ImageOutputFeature
    schema_cls = ImageOutputFeatureConfig
    output_config = schema_cls.from_dict(output_feature_def)
    output_feature_obj = feature_cls(output_config, {}).to(DEVICE)

    combiner_outputs = {
        "combiner_output": hidden,
        ENCODER_OUTPUT_STATE: encoder_output[ENCODER_OUTPUT_STATE],
    }

    image_output = output_feature_obj(combiner_outputs, {})

    assert LOGITS in image_output
    assert image_output[LOGITS].size() == torch.Size([BATCH_SIZE, num_classes, height, width])


def test_image_preproc_module_bad_num_channels():
    metadata = {
        "preprocessing": {
            "missing_value_strategy": BFILL,
            "in_memory": True,
            "resize_method": "interpolate",
            "scaling": "pixel_normalization",
            "num_processes": 1,
            "infer_image_num_channels": True,
            "infer_image_dimensions": True,
            "infer_image_max_height": 256,
            "infer_image_max_width": 256,
            "infer_image_sample_size": 100,
            "height": 12,
            "width": 12,
            "num_channels": 2,
            "num_classes": 0,
            "channel_class_map": [],
        },
        "reshape": (2, 12, 12),
    }
    module = _ImagePreprocessing(metadata)

    with pytest.raises(ValueError):
        module(torch.rand(2, 3, 10, 10))


@pytest.mark.parametrize("resize_method", [INTERPOLATE, CROP_OR_PAD])
@pytest.mark.parametrize(["num_channels", "num_channels_expected"], [(1, 3), (3, 1)])
def test_image_preproc_module_list_of_tensors(resize_method, num_channels, num_channels_expected):
    metadata = {
        "preprocessing": {
            "missing_value_strategy": BFILL,
            "in_memory": True,
            "resize_method": resize_method,
            "scaling": "pixel_normalization",
            "num_processes": 1,
            "infer_image_num_channels": True,
            "infer_image_dimensions": True,
            "infer_image_max_height": 256,
            "infer_image_max_width": 256,
            "infer_image_sample_size": 100,
            "height": 12,
            "width": 12,
            "num_channels": num_channels_expected,
            "num_classes": 0,
            "channel_class_map": [],
        },
        "reshape": (num_channels_expected, 12, 12),
    }
    module = _ImagePreprocessing(metadata)

    res = module([torch.rand(num_channels, 25, 25), torch.rand(num_channels, 10, 10)])

    assert res.shape == torch.Size((2, num_channels_expected, 12, 12))


@pytest.mark.parametrize("resize_method", [INTERPOLATE, CROP_OR_PAD])
@pytest.mark.parametrize(["num_channels", "num_channels_expected"], [(1, 3), (3, 1)])
def test_image_preproc_module_tensor(resize_method, num_channels, num_channels_expected):
    metadata = {
        "preprocessing": {
            "missing_value_strategy": BFILL,
            "in_memory": True,
            "resize_method": resize_method,
            "scaling": "pixel_normalization",
            "num_processes": 1,
            "infer_image_num_channels": True,
            "infer_image_dimensions": True,
            "infer_image_max_height": 256,
            "infer_image_max_width": 256,
            "infer_image_sample_size": 100,
            "height": 12,
            "width": 12,
            "num_channels": num_channels_expected,
            "num_classes": 0,
            "channel_class_map": [],
        },
        "reshape": (num_channels_expected, 12, 12),
    }
    module = _ImagePreprocessing(metadata)

    res = module(torch.rand(2, num_channels, 10, 10))

    assert res.shape == torch.Size((2, num_channels_expected, 12, 12))


@pytest.mark.parametrize(["height", "width"], [(224, 224), (32, 32)])
def test_image_preproc_module_class_map(height, width):
    metadata = {
        "preprocessing": {
            "num_processes": 1,
            "resize_method": CROP_OR_PAD,
            "infer_image_num_channels": True,
            "infer_image_dimensions": True,
            "infer_image_max_height": height,
            "infer_image_max_width": width,
            "infer_image_sample_size": 100,
            "infer_image_num_classes": True,
            "height": height,
            "width": width,
            "num_channels": 3,
            "num_classes": 8,
            "channel_class_map": [
                [40, 40, 40],
                [40, 40, 41],
                [40, 41, 40],
                [40, 41, 41],
                [41, 40, 40],
                [41, 40, 41],
                [41, 41, 40],
                [41, 41, 41],
            ],
        },
    }
    module = _ImagePreprocessing(metadata)

    res = module(torch.randint(40, 42, (2, 3, height, width)))

    assert res.shape == torch.Size((2, height, width))
    assert torch.all(res.ge(0)) and torch.all(res.le(7))
