from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.constants import CROP_OR_PAD, INTERPOLATE
from ludwig.features.image_feature import _ImagePreprocessing, ImageInputFeature
from ludwig.models.ecd import build_single_input

BATCH_SIZE = 2


@pytest.fixture(scope="module")
def image_config():
    return {
        "name": "image_column_name",
        "type": "image",
        "encoder": "stacked_cnn",
        "tied": None,
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
        "scaling": "pixel_normalization",
        "preprocessing": {"height": 28, "width": 28, "num_channels": 1},  # example pre-processing
    }


@pytest.mark.parametrize(
    "encoder, height, width, num_channels",
    [
        ("resnet", 28, 28, 3),
        ("stacked_cnn", 28, 28, 3),
        ("stacked_cnn", 28, 28, 1),
        ("mlp_mixer", 32, 32, 3),
        ("vit", 224, 224, 3),
    ],
)
def test_image_input_feature(image_config: Dict, encoder: str, height: int, width: int, num_channels) -> None:
    # setup image input feature definition
    image_def = deepcopy(image_config)
    image_def["encoder"] = encoder
    image_def["height"] = height
    image_def["width"] = width
    image_def["num_channels"] = num_channels

    # pickup any other missing parameters
    ImageInputFeature.populate_defaults(image_def)

    # ensure no exceptions raised during build
    input_feature_obj = build_single_input(image_def, None)

    # check one forward pass through input feature
    input_tensor = torch.randint(0, 256, size=(BATCH_SIZE, num_channels, height, width), dtype=torch.uint8)

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)

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


def test_image_preproc_module_bad_num_channels():
    metadata = {
        "preprocessing": {
            "missing_value_strategy": "backfill",
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
            "missing_value_strategy": "backfill",
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
            "missing_value_strategy": "backfill",
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
        },
        "reshape": (num_channels_expected, 12, 12),
    }
    module = _ImagePreprocessing(metadata)

    res = module(torch.rand(2, num_channels, 10, 10))

    assert res.shape == torch.Size((2, num_channels_expected, 12, 12))
