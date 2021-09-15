import pytest
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from ludwig.modules.convolutional_modules import Conv2DLayer, Conv2DStack,\
    Conv2DLayerFixedPadding, ResNetBlock, ResNetBottleneckBlock,\
    ResNetBlockLayer, ResNet


@pytest.mark.parametrize(
    ('img_height,img_width,in_channels,out_channels,pool_kernel_size,'
     'pool_stride,pool_padding,pool_dilation'),
    [(224, 224, 3, 16, 2, 2, 0, 1)]
)
@pytest.mark.parametrize(
    'stride,padding',
    [(1, 'valid'), (1, 'same'), (2, 'valid')]
)
@pytest.mark.parametrize('kernel_size', [1, 3, 5])
@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('norm', ['batch', 'layer'])
def test_conv2d_layer(
        img_height: int,
        img_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, Tuple[int], str],
        dilation: Union[int, Tuple[int]],
        norm: str,
        pool_kernel_size: Union[int, Tuple[int]],
        pool_stride: int,
        pool_padding: Union[int, Tuple[int], str],
        pool_dilation: Union[int, Tuple[int]]
) -> None:
    conv2d_layer = Conv2DLayer(
        img_height=img_height, img_width=img_width, in_channels=in_channels,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, norm=norm,
        pool_kernel_size=pool_kernel_size, pool_stride=pool_stride,
        pool_padding=pool_padding, pool_dilation=pool_dilation
    )
    input_tensor = torch.rand(2, in_channels, img_height, img_width)
    output_tensor = conv2d_layer(input_tensor)
    assert output_tensor.shape[1:] == conv2d_layer.output_shape


@pytest.mark.parametrize('img_height,img_width', [(224, 224)])
@pytest.mark.parametrize(
    'layers,num_layers,first_in_channels',
    [
        (None, None, 3),
        (None, 5, 3),
        ([{'out_channels': 8}], None, 3),
        ([{'out_channels': 8, 'in_channels': 3}], None, None)
    ]
)
def test_conv2d_stack(
        img_height: int,
        img_width: int,
        layers: Optional[List[Dict]],
        num_layers: Optional[int],
        first_in_channels: Optional[int]
) -> None:
    conv2d_stack = Conv2DStack(
        img_height=img_height, img_width=img_width, layers=layers,
        num_layers=num_layers, first_in_channels=first_in_channels
    )
    input_tensor = torch.rand(2, 3, img_height, img_width)
    output_tensor = conv2d_stack(input_tensor)
    assert output_tensor.shape[1:] == conv2d_stack.output_shape


@pytest.mark.parametrize('img_height,img_width,in_channels', [(224, 224, 8)])
@pytest.mark.parametrize('stride', [1, 3])
@pytest.mark.parametrize('groups', [1, 8])
def test_conv2d_layer_fixed_padding(
        img_height: int,
        img_width: int,
        in_channels: int,
        stride: int,
        groups: int
) -> None:
    conv2d_fixed_padding = Conv2DLayerFixedPadding(
        img_height=img_height, img_width=img_width, in_channels=in_channels,
        stride=stride, groups=groups
    )
    input_tensor = torch.rand(2, in_channels, img_height, img_width)
    output_tensor = conv2d_fixed_padding(input_tensor)
    assert output_tensor.shape[1:] == conv2d_fixed_padding.output_shape


@pytest.mark.parametrize(
    'img_height,img_width,first_in_channels,out_channels',
    [(224, 224, 64, 64)]
)
@pytest.mark.parametrize(
    'projection_shortcut',
    [   
        None,
        Conv2DLayerFixedPadding(
            img_height=224, img_width=224, in_channels=64, out_channels=64
        )
    ]
)
def test_resnet_block(
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        projection_shortcut: Callable
) -> None:
    resnet_block = ResNetBlock(
        img_height=img_height, img_width=img_width,
        first_in_channels=first_in_channels, out_channels=out_channels,
        projection_shortcut=projection_shortcut
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block(input_tensor)
    assert output_tensor.shape[1:] == resnet_block.output_shape


@pytest.mark.parametrize(
    'img_height,img_width,first_in_channels,out_channels',
    [(224, 224, 64, 64)]
)
@pytest.mark.parametrize(
    'projection_shortcut',
    [   
        None,
        Conv2DLayerFixedPadding(
            img_height=224, img_width=224, in_channels=64, out_channels=256
        )
    ]
)
def test_resnet_bottleneck_block(
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        projection_shortcut: Callable
) -> None:
    resnet_block = ResNetBottleneckBlock(
        img_height=img_height, img_width=img_width,
        first_in_channels=first_in_channels, out_channels=out_channels,
        projection_shortcut=projection_shortcut
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block(input_tensor)
    assert output_tensor.shape[1:] == resnet_block.output_shape


@pytest.mark.parametrize(
    'img_height,img_width,first_in_channels,out_channels,num_blocks',
    [(224, 224, 3, 64, 3)]
)
@pytest.mark.parametrize(
    'is_bottleneck, block_fn',
    [(True, ResNetBottleneckBlock), (False, ResNetBlock)]
)
def test_resnet_block_layer(
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        is_bottleneck: bool,
        block_fn: Union[ResNetBlock, ResNetBottleneckBlock],
        num_blocks: int,
):
    resnet_block_layer = ResNetBlockLayer(
        img_height=img_height, img_width=img_width,
        first_in_channels=first_in_channels, out_channels=out_channels,
        is_bottleneck=is_bottleneck, block_fn=block_fn, num_blocks=num_blocks
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block_layer(input_tensor)
    assert output_tensor.shape[1:] == resnet_block_layer.output_shape


@pytest.mark.parametrize(
    'img_height,img_width,first_in_channels,out_channels', [(224, 224, 3, 64)]
)
@pytest.mark.parametrize('resnet_size', [18, 34, 50])
def test_resnet(
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        resnet_size: int,
):
    resnet = ResNet(
        img_height=img_height, img_width=img_width,
        first_in_channels=first_in_channels, out_channels=out_channels,
        resnet_size=resnet_size
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet(input_tensor)
    assert output_tensor.shape[1:] == resnet.output_shape
