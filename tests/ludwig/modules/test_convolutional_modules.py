from typing import Callable, Dict, List, Optional, Tuple, Union

import pytest
import torch

from ludwig.modules.convolutional_modules import (
    Conv1DLayer,
    Conv1DStack,
    Conv2DLayer,
    Conv2DLayerFixedPadding,
    Conv2DStack,
    ParallelConv1D,
    ParallelConv1DStack,
    ResNet,
    ResNetBlock,
    ResNetBlockLayer,
    ResNetBottleneckBlock,
)
from ludwig.utils.image_utils import get_img_output_shape
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

BATCH_SIZE = 2
SEQ_SIZE = 17
HIDDEN_SIZE = 8
NUM_FILTERS = 4

RANDOM_SEED = 1919


###
# Helper function to compute expected output shape
# for Conv1D related layers
###
def expected_seq_size(
    seq_size: int,  # input max sequence length
    padding: str,  # conv1d padding: 'same' or 'valid'
    kernel_size: int,  # conv1d kernel size
    stride: int,  # conv1d stride
    dilation: int,  # conv1d dilation rate
    pool_size: Union[None, int],  # pooling layer kernel size
    pool_padding: str,  # pooling layer padding: 'same' or 'valid'
    pool_stride: int,  # pooling layer stride
) -> int:
    # output shape for the convolutional layer
    output_seq_size = get_img_output_shape(
        img_height=0,  # img_height set to zero for 1D structure
        img_width=seq_size,  # img_width equates to max sequence length
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if pool_size is not None:
        # pooling layer present, adjust expected output shape for pooling layer
        output_seq_size = get_img_output_shape(
            img_height=0,  # img_height set to zero for 1D structure
            img_width=output_seq_size[1],  # img_width equates to max sequence length
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
            dilation=1,  # pooling layer only support unit dilation
        )
    return output_seq_size[1]


###
# 1D Convolutional Tests
###
@pytest.mark.parametrize("pool_function", ["max", "mean"])
@pytest.mark.parametrize(
    "pool_size, pool_padding, pool_stride",
    [(None, None, None), (3, "same", 1), (5, "same", 1), (3, "valid", 2), (5, "valid", 2)],
)
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("strides, padding", [(1, "same"), (1, "valid"), (2, "valid")])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_conv1d_layer(
    kernel_size: int,
    strides: int,
    padding: str,
    dilation: int,
    pool_size: Union[None, int],
    pool_padding: str,
    pool_stride: int,
    pool_function: str,
) -> None:
    # make test repeatable
    torch.manual_seed(RANDOM_SEED)

    # setup synthetic tensor for test
    input = torch.randn([BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    conv1_layer = Conv1DLayer(
        in_channels=HIDDEN_SIZE,
        out_channels=NUM_FILTERS,
        max_sequence_length=SEQ_SIZE,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        pool_function=pool_function,
        pool_size=pool_size,
        pool_strides=pool_stride,
        pool_padding=pool_padding,
    )

    out_tensor = conv1_layer(input)

    # check for correct output class
    assert isinstance(out_tensor, torch.Tensor)

    # check for correct output shape
    output_seq_size = expected_seq_size(
        seq_size=SEQ_SIZE,
        padding=padding,
        kernel_size=kernel_size,
        stride=strides,
        dilation=dilation,
        pool_size=pool_size,
        pool_padding=pool_padding,
        pool_stride=pool_stride,
    )
    assert out_tensor.size() == (BATCH_SIZE, output_seq_size, NUM_FILTERS)


@pytest.mark.parametrize("dropout", [0, 0.5])
@pytest.mark.parametrize(
    "layers, num_layers",
    [
        (None, None),  # setup up default number of layers with default values
        (None, 4),  # setup of 4 layers with default values
        ([{"num_filters": NUM_FILTERS - 2}, {"num_filters": NUM_FILTERS + 2}], None),  # 2 custom layers
    ],
)
def test_conv1d_stack(layers: Union[None, list], num_layers: Union[None, int], dropout: float) -> None:
    # make test repeatable
    torch.manual_seed(RANDOM_SEED)

    # setup synthetic input tensor for test
    input = torch.randn([BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    conv1_stack = Conv1DStack(
        in_channels=HIDDEN_SIZE,
        out_channels=NUM_FILTERS,
        max_sequence_length=SEQ_SIZE,
        layers=layers,
        num_layers=num_layers,
        default_num_filters=NUM_FILTERS,
        default_dropout=dropout,
    )

    # check for correct stack formation
    if layers is None:
        assert len(conv1_stack.stack) == 6 if num_layers is None else num_layers
    else:
        # custom layer specification
        assert len(conv1_stack.stack) == len(layers)
        assert conv1_stack.stack[0].out_channels == NUM_FILTERS - 2
        assert conv1_stack.stack[1].out_channels == NUM_FILTERS + 2

    # generate output tensor
    out_tensor = conv1_stack(input)

    # check for correct output class
    assert isinstance(out_tensor, torch.Tensor)

    assert out_tensor.size()[1:] == conv1_stack.output_shape[:]

    # check for correct output shape
    last_module = conv1_stack.stack[-1]
    output_seq_size = expected_seq_size(
        seq_size=last_module.input_shape[0],
        padding=last_module.padding,
        kernel_size=last_module.kernel_size,
        stride=last_module.stride,
        dilation=last_module.dilation,
        pool_size=last_module.pool_size,
        pool_padding=last_module.pool_padding,
        pool_stride=last_module.pool_strides,
    )
    if layers is None:
        # default stack setup
        assert out_tensor.size() == (BATCH_SIZE, output_seq_size, NUM_FILTERS)
    else:
        # custom stack setup
        assert out_tensor.size() == (BATCH_SIZE, output_seq_size, NUM_FILTERS + 2)

    # check for parameter updates
    target = torch.randn(conv1_stack.output_shape)
    _, tpc, upc, not_updated = check_module_parameters_updated(conv1_stack, (input,), target)
    if dropout == 0:
        # all trainable parameters should be updated
        assert tpc == upc, (
            f"All parameter not updated. Parameters not updated: {not_updated}" f"\nModule structure:\n{conv1_stack}"
        )
    else:
        # with specified config and random seed, non-zero dropout update parameter count could take different values
        assert (tpc == upc) or (upc == 1), (
            f"All parameter not updated. Parameters not updated: {not_updated}" f"\nModule structure:\n{conv1_stack}"
        )


@pytest.mark.parametrize(
    "layers",
    [
        None,  # setup up default number of layers with default values
        [{"filter_size": 3}, {"filter_size": 4}],  # custom parallel layers
    ],
)
def test_parallel_conv1d(layers: Union[None, list]) -> None:
    input = torch.randn([BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    parallel_conv1d = ParallelConv1D(
        in_channels=HIDDEN_SIZE,
        out_channels=NUM_FILTERS,
        max_sequence_length=SEQ_SIZE,
        layers=layers,
        default_num_filters=NUM_FILTERS,
    )

    # check for correct stack formation
    if layers is None:
        assert len(parallel_conv1d.parallel_layers) == 4
    else:
        # custom layer specification
        assert len(parallel_conv1d.parallel_layers) == len(layers)
        assert parallel_conv1d.parallel_layers[0].kernel_size == 3
        assert parallel_conv1d.parallel_layers[1].kernel_size == 4

    # generate output tensor
    out_tensor = parallel_conv1d(input)

    # check for correct output class
    assert isinstance(out_tensor, torch.Tensor)

    # check for correct output shape
    parallel_module = parallel_conv1d.parallel_layers[0]
    output_seq_size = expected_seq_size(
        seq_size=parallel_module.input_shape[0],
        padding=parallel_module.padding,
        kernel_size=parallel_module.kernel_size,
        stride=parallel_module.stride,
        dilation=parallel_module.dilation,
        pool_size=parallel_module.pool_size,
        pool_padding=parallel_module.pool_padding,
        pool_stride=parallel_module.pool_strides,
    )

    assert out_tensor.size() == (BATCH_SIZE, output_seq_size, len(parallel_conv1d.parallel_layers) * NUM_FILTERS)


TEST_FILTER_SIZE0 = 7
TEST_FILTER_SIZE1 = 5


@pytest.mark.parametrize("dropout", [0, 0.99])
@pytest.mark.parametrize(
    "stacked_layers",
    [
        None,  # setup up default number of layers with default values
        # custom stacked parallel layers
        [
            [  # parallel_conv1d_stack.stack[0]
                {"filter_size": 3},
                {"filter_size": 5},
                {"filter_size": TEST_FILTER_SIZE0},
            ],
            [  # parallel_conv1d_stack.stack[1]
                {"filter_size": 2},
                {"filter_size": 3},
                {"filter_size": 4},
                {"filter_size": TEST_FILTER_SIZE1},
            ],
        ],
    ],
)
def test_parallel_conv1d_stack(stacked_layers: Union[None, list], dropout: float) -> None:
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    # setup synthetic input tensor for test
    input = torch.randn([BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    parallel_conv1d_stack = ParallelConv1DStack(
        in_channels=HIDDEN_SIZE,
        out_channels=NUM_FILTERS,
        max_sequence_length=SEQ_SIZE,
        stacked_layers=stacked_layers,
        default_num_filters=NUM_FILTERS,
        default_dropout=dropout,
    )

    # check for correct stack formation
    if stacked_layers is None:
        assert len(parallel_conv1d_stack.stack) == 3
        for i in range(len(parallel_conv1d_stack.stack)):
            assert len(parallel_conv1d_stack.stack[i].parallel_layers) == 4
    else:
        # spot check custom layer specification
        assert len(parallel_conv1d_stack.stack) == len(stacked_layers)
        assert len(parallel_conv1d_stack.stack[0].parallel_layers) == 3
        assert parallel_conv1d_stack.stack[0].parallel_layers[2].kernel_size == TEST_FILTER_SIZE0
        assert len(parallel_conv1d_stack.stack[1].parallel_layers) == 4
        assert parallel_conv1d_stack.stack[1].parallel_layers[3].kernel_size == TEST_FILTER_SIZE1

    # generate output tensor
    out_tensor = parallel_conv1d_stack(input)

    # check for correct output class
    assert isinstance(out_tensor, torch.Tensor)

    # check output shape
    assert out_tensor.size() == (BATCH_SIZE, *parallel_conv1d_stack.output_shape)

    # check for parameter updates
    target = torch.randn(parallel_conv1d_stack.output_shape)
    _, tpc, upc, not_updated = check_module_parameters_updated(parallel_conv1d_stack, (input,), target)
    if dropout == 0:
        # all trainable parameters should be updated
        assert tpc == upc, (
            f"All parameter not updated. Parameters not updated: {not_updated}"
            f"\nModule structure:\n{parallel_conv1d_stack}"
        )
    else:
        # with specified config and random seed, non-zero dropout update parameter count could take different values
        assert (tpc == upc) or (upc == 5), (
            f"All parameter not updated. Parameters not updated: {not_updated}"
            f"\nModule structure:\n{parallel_conv1d_stack}"
        )


###
#  2D Convolutional Tests
###
@pytest.mark.parametrize(
    ("img_height,img_width,in_channels,out_channels,pool_kernel_size," "pool_stride,pool_padding,pool_dilation"),
    [(224, 224, 3, 16, 2, 2, 0, 1)],
)
@pytest.mark.parametrize("stride,padding", [(1, "valid"), (1, "same"), (2, "valid")])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("norm", ["batch", "layer"])
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
    pool_dilation: Union[int, Tuple[int]],
) -> None:
    conv2d_layer = Conv2DLayer(
        img_height=img_height,
        img_width=img_width,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        norm=norm,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        pool_dilation=pool_dilation,
    )
    input_tensor = torch.rand(2, in_channels, img_height, img_width)
    output_tensor = conv2d_layer(input_tensor)
    assert output_tensor.shape[1:] == conv2d_layer.output_shape


@pytest.mark.parametrize("img_height,img_width", [(224, 224)])
@pytest.mark.parametrize(
    "layers,num_layers,first_in_channels",
    [
        (None, None, 3),
        (None, 5, 3),
        ([{"out_channels": 8}], None, 3),
        ([{"out_channels": 8, "in_channels": 3}], None, None),
    ],
)
def test_conv2d_stack(
    img_height: int,
    img_width: int,
    layers: Optional[List[Dict]],
    num_layers: Optional[int],
    first_in_channels: Optional[int],
) -> None:
    conv2d_stack = Conv2DStack(
        img_height=img_height,
        img_width=img_width,
        layers=layers,
        num_layers=num_layers,
        first_in_channels=first_in_channels,
    )
    input_tensor = torch.rand(2, 3, img_height, img_width)
    output_tensor = conv2d_stack(input_tensor)
    assert output_tensor.shape[1:] == conv2d_stack.output_shape


@pytest.mark.parametrize("img_height,img_width,in_channels", [(224, 224, 8)])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("groups", [1, 8])
def test_conv2d_layer_fixed_padding(
    img_height: int, img_width: int, in_channels: int, stride: int, groups: int
) -> None:
    conv2d_fixed_padding = Conv2DLayerFixedPadding(
        img_height=img_height, img_width=img_width, in_channels=in_channels, stride=stride, groups=groups
    )
    input_tensor = torch.rand(2, in_channels, img_height, img_width)
    output_tensor = conv2d_fixed_padding(input_tensor)
    assert output_tensor.shape[1:] == conv2d_fixed_padding.output_shape


@pytest.mark.parametrize("img_height,img_width,first_in_channels,out_channels", [(224, 224, 64, 64)])
@pytest.mark.parametrize(
    "projection_shortcut",
    [None, Conv2DLayerFixedPadding(img_height=224, img_width=224, in_channels=64, out_channels=64)],
)
def test_resnet_block(
    img_height: int, img_width: int, first_in_channels: int, out_channels: int, projection_shortcut: Callable
) -> None:
    resnet_block = ResNetBlock(
        img_height=img_height,
        img_width=img_width,
        first_in_channels=first_in_channels,
        out_channels=out_channels,
        projection_shortcut=projection_shortcut,
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block(input_tensor)
    assert output_tensor.shape[1:] == resnet_block.output_shape


@pytest.mark.parametrize("img_height,img_width,first_in_channels,out_channels", [(224, 224, 64, 64)])
@pytest.mark.parametrize(
    "projection_shortcut",
    [None, Conv2DLayerFixedPadding(img_height=224, img_width=224, in_channels=64, out_channels=256)],
)
def test_resnet_bottleneck_block(
    img_height: int, img_width: int, first_in_channels: int, out_channels: int, projection_shortcut: Callable
) -> None:
    resnet_block = ResNetBottleneckBlock(
        img_height=img_height,
        img_width=img_width,
        first_in_channels=first_in_channels,
        out_channels=out_channels,
        projection_shortcut=projection_shortcut,
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block(input_tensor)
    assert output_tensor.shape[1:] == resnet_block.output_shape


@pytest.mark.parametrize("img_height,img_width,first_in_channels,out_channels,num_blocks", [(224, 224, 3, 32, 3)])
@pytest.mark.parametrize("is_bottleneck, block_fn", [(True, ResNetBottleneckBlock), (False, ResNetBlock)])
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
        img_height=img_height,
        img_width=img_width,
        first_in_channels=first_in_channels,
        out_channels=out_channels,
        is_bottleneck=is_bottleneck,
        block_fn=block_fn,
        num_blocks=num_blocks,
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet_block_layer(input_tensor)
    assert output_tensor.shape[1:] == resnet_block_layer.output_shape


@pytest.mark.parametrize("img_height,img_width,first_in_channels,out_channels", [(224, 224, 3, 64)])
@pytest.mark.parametrize("resnet_size", [18, 34, 50])
def test_resnet(
    img_height: int,
    img_width: int,
    first_in_channels: int,
    out_channels: int,
    resnet_size: int,
):
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    resnet = ResNet(
        img_height=img_height,
        img_width=img_width,
        first_in_channels=first_in_channels,
        out_channels=out_channels,
        resnet_size=resnet_size,
    )
    input_tensor = torch.rand(2, first_in_channels, img_height, img_width)
    output_tensor = resnet(input_tensor)
    assert output_tensor.shape[1:] == resnet.output_shape

    # check for parameter updates
    target = torch.randn(output_tensor.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(resnet, (input_tensor,), target)
    # all trainable parameters should be updated
    assert tpc == upc, (
        f"All parameter not updated. Parameters not updated: {not_updated}" f"\nModule structure:\n{resnet}"
    )
