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
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ludwig.utils.image_utils import get_img_output_shape
from ludwig.utils.torch_utils import get_activation, LudwigModule

logger = logging.getLogger(__name__)


class Conv1DLayer(LudwigModule):
    def __init__(
        self,
        in_channels=1,
        out_channels=256,
        max_sequence_length=None,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation=1,
        groups=1,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        activation="relu",
        dropout=0,
        pool_function="max",
        pool_size=2,
        pool_strides=None,
        pool_padding="valid",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_sequence_length = max_sequence_length
        self.kernel_size = kernel_size
        self.stride = strides
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.pool_size = pool_size
        if pool_strides is None:
            self.pool_strides = pool_size
        else:
            self.pool_strides = pool_strides
        if pool_padding == "same" and pool_size is not None:
            self.pool_padding = (self.pool_size - 1) // 2
        else:
            self.pool_padding = 0

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,),
                stride=(strides,),
                padding=padding,
                dilation=(dilation,),
            )
        )

        if norm and norm_params is None:
            norm_params = {}
        if norm == "batch":
            self.layers.append(nn.BatchNorm1d(num_features=out_channels, **norm_params))
        elif norm == "layer":
            self.layers.append(nn.LayerNorm(normalized_shape=[out_channels, self.max_sequence_length], **norm_params))

        self.layers.append(get_activation(activation))

        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        if pool_size is not None:
            pool = nn.MaxPool1d
            if pool_function in {"average", "avg", "mean"}:
                pool = nn.AvgPool1d
            self.layers.append(pool(kernel_size=self.pool_size, stride=self.pool_strides, padding=self.pool_padding))

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

    @property
    def input_shape(self):
        """Returns the size of the input tensor without the batch dimension."""
        return torch.Size([self.max_sequence_length, self.in_channels])

    def forward(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_size, in_channels]
        # in Torch nomenclature (N, L, C)
        hidden = inputs

        # put in torch compatible form [batch_size, in_channels, seq_size]
        hidden = hidden.transpose(1, 2)

        for layer in self.layers:
            hidden = layer(hidden)

        # revert back to normal form [batch_size, seq_size, out_channels]
        hidden = hidden.transpose(1, 2)

        return hidden  # (batch_size, seq_size, out_channels)


class Conv1DStack(LudwigModule):
    def __init__(
        self,
        in_channels=1,
        max_sequence_length=None,
        layers=None,
        num_layers=None,
        default_num_filters=256,
        default_filter_size=3,
        default_strides=1,
        default_padding="same",
        default_dilation_rate=1,
        default_use_bias=True,
        default_weights_initializer="xavier_uniform",
        default_bias_initializer="zeros",
        default_norm=None,
        default_norm_params=None,
        default_activation="relu",
        default_dropout=0,
        default_pool_function="max",
        default_pool_size=2,
        default_pool_strides=None,
        default_pool_padding="same",
        **kwargs,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.in_channels = in_channels

        if layers is None:
            if num_layers is None:
                self.layers = [
                    {"filter_size": 7, "pool_size": 3},
                    {"filter_size": 7, "pool_size": 3},
                    {"filter_size": 3, "pool_size": None},
                    {"filter_size": 3, "pool_size": None},
                    {"filter_size": 3, "pool_size": None},
                    {"filter_size": 3, "pool_size": 3},
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append(
                        {
                            "filter_size": default_filter_size,
                            "num_filters": default_num_filters,
                            "pool_size": default_pool_size,
                            "pool_strides": default_pool_strides,
                        }
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if "num_filters" not in layer:
                layer["num_filters"] = default_num_filters
            if "filter_size" not in layer:
                layer["filter_size"] = default_filter_size
            if "strides" not in layer:
                layer["strides"] = default_strides
            if "padding" not in layer:
                layer["padding"] = default_padding
            if "dilation_rate" not in layer:
                layer["dilation_rate"] = default_dilation_rate
            if "use_bias" not in layer:
                layer["use_bias"] = default_use_bias
            if "weights_initializer" not in layer:
                layer["weights_initializer"] = default_weights_initializer
            if "bias_initializer" not in layer:
                layer["bias_initializer"] = default_bias_initializer
            if "norm" not in layer:
                layer["norm"] = default_norm
            if "norm_params" not in layer:
                layer["norm_params"] = default_norm_params
            if "activation" not in layer:
                layer["activation"] = default_activation
            if "dropout" not in layer:
                layer["dropout"] = default_dropout
            if "pool_function" not in layer:
                layer["pool_function"] = default_pool_function
            if "pool_size" not in layer:
                layer["pool_size"] = default_pool_size
            if "pool_strides" not in layer:
                layer["pool_strides"] = default_pool_strides
            if "pool_padding" not in layer:
                layer["pool_padding"] = default_pool_padding

        self.stack = nn.ModuleList()

        prior_layer_channels = in_channels
        l_in = self.max_sequence_length  # torch L_in
        for i, layer in enumerate(self.layers):
            logger.debug(f"   stack layer {i}")
            self.stack.append(
                Conv1DLayer(
                    in_channels=prior_layer_channels,
                    out_channels=layer["num_filters"],
                    max_sequence_length=l_in,
                    kernel_size=layer["filter_size"],
                    strides=layer["strides"],
                    padding=layer["padding"],
                    dilation=layer["dilation_rate"],
                    use_bias=layer["use_bias"],
                    weights_initializer=layer["weights_initializer"],
                    bias_initializer=layer["bias_initializer"],
                    norm=layer["norm"],
                    norm_params=layer["norm_params"],
                    activation=layer["activation"],
                    dropout=layer["dropout"],
                    pool_function=layer["pool_function"],
                    pool_size=layer["pool_size"],
                    pool_strides=layer["pool_strides"],
                    pool_padding=layer["pool_padding"],
                )
            )

            # retrieve number of channels from prior layer
            input_shape = self.stack[i].input_shape
            output_shape = self.stack[i].output_shape

            logger.debug(f"{self.__class__.__name__}: " f"input_shape {input_shape}, output shape {output_shape}")

            # pass along shape for the input to the next layer
            l_in, prior_layer_channels = output_shape

    @property
    def input_shape(self):
        """Returns the size of the input tensor without the batch dimension."""
        return torch.Size([self.max_sequence_length, self.in_channels])

    def forward(self, inputs, mask=None):
        hidden = inputs

        # todo: enumerate for debugging, remove after testing
        for i, layer in enumerate(self.stack):
            hidden = layer(hidden)

        if hidden.shape[1] == 0:
            raise ValueError(
                "The output of the conv stack has the second dimension "
                "(length of the sequence) equal to 0. "
                "This means that the combination of filter_size, padding, "
                "stride, pool_size, pool_padding and pool_stride reduces "
                "the sequence length more than is possible. "
                'Try using "same" padding and reducing or eliminating stride '
                "and pool."
            )

        return hidden


class ParallelConv1D(LudwigModule):
    def __init__(
        self,
        in_channels=1,
        max_sequence_length=None,
        layers=None,
        default_num_filters=256,
        default_filter_size=3,
        default_strides=1,
        default_padding="same",
        default_dilation_rate=1,
        default_use_bias=True,
        default_weights_initializer="xavier_uniform",
        default_bias_initializer="zeros",
        default_norm=None,
        default_norm_params=None,
        default_activation="relu",
        default_dropout=0,
        default_pool_function="max",
        default_pool_size=None,
        default_pool_strides=None,
        default_pool_padding="valid",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.max_sequence_length = max_sequence_length

        if layers is None:
            self.layers = [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}]
        else:
            self.layers = layers

        for layer in self.layers:
            if "num_filters" not in layer:
                layer["num_filters"] = default_num_filters
            if "filter_size" not in layer:
                layer["filter_size"] = default_filter_size
            if "strides" not in layer:
                layer["strides"] = default_strides
            if "padding" not in layer:
                layer["padding"] = default_padding
            if "dilation_rate" not in layer:
                layer["dilation_rate"] = default_dilation_rate
            if "use_bias" not in layer:
                layer["use_bias"] = default_use_bias
            if "weights_initializer" not in layer:
                layer["weights_initializer"] = default_weights_initializer
            if "bias_initializer" not in layer:
                layer["bias_initializer"] = default_bias_initializer
            if "norm" not in layer:
                layer["norm"] = default_norm
            if "norm_params" not in layer:
                layer["norm_params"] = default_norm_params
            if "activation" not in layer:
                layer["activation"] = default_activation
            if "dropout" not in layer:
                layer["dropout"] = default_dropout
            if "pool_function" not in layer:
                layer["pool_function"] = default_pool_function
            if "pool_size" not in layer:
                layer["pool_size"] = default_pool_size
            if "pool_strides" not in layer:
                layer["pool_strides"] = default_pool_strides
            if "pool_padding" not in layer:
                layer["pool_padding"] = default_pool_padding

        self.parallel_layers = nn.ModuleList()

        for i, layer in enumerate(self.layers):
            logger.debug(f"   parallel layer {i}")
            self.parallel_layers.append(
                Conv1DLayer(
                    in_channels=self.in_channels,
                    out_channels=layer["num_filters"],
                    max_sequence_length=self.max_sequence_length,
                    kernel_size=layer["filter_size"],
                    strides=layer["strides"],
                    padding=layer["padding"],
                    dilation=layer["dilation_rate"],
                    use_bias=layer["use_bias"],
                    weights_initializer=layer["weights_initializer"],
                    bias_initializer=layer["bias_initializer"],
                    norm=layer["norm"],
                    norm_params=layer["norm_params"],
                    activation=layer["activation"],
                    dropout=layer["dropout"],
                    pool_function=layer["pool_function"],
                    pool_size=layer["pool_size"],
                    pool_strides=layer["pool_strides"],
                    pool_padding=layer["pool_padding"],
                )
            )

            logger.debug(
                f"{self.__class__.__name__} layer {i}, input shape "
                f"{self.parallel_layers[i].input_shape}, output shape "
                f"{self.parallel_layers[i].output_shape}"
            )

    @property
    def input_shape(self) -> torch.Size:
        """Returns the size of the input tensor without the batch dimension."""
        return torch.Size([self.max_sequence_length, self.in_channels])

    def forward(self, inputs, mask=None):
        # inputs: [batch_size, seq_size, in_channels)

        hidden = inputs
        hiddens = []

        for layer in self.parallel_layers:
            hiddens.append(layer(hidden))
        hidden = torch.cat(hiddens, 2)

        if hidden.shape[1] == 0:
            raise ValueError(
                "The output of the conv stack has the second dimension "
                "(length of the sequence) equal to 0. "
                "This means that the combination of filter_size, padding, "
                "stride, pool_size, pool_padding and pool_stride reduces "
                "the sequence length more than is possible. "
                'Try using "same" padding and reducing or eliminating stride '
                "and pool."
            )

        # (batch_size, seq_size, len(parallel_layers) * out_channels)
        return hidden


class ParallelConv1DStack(LudwigModule):
    def __init__(
        self,
        in_channels=None,
        stacked_layers=None,
        max_sequence_length=None,
        default_num_filters=64,
        default_filter_size=3,
        default_strides=1,
        default_padding="same",
        default_dilation_rate=1,
        default_use_bias=True,
        default_weights_initializer="xavier_uniform",
        default_bias_initializer="zeros",
        default_norm=None,
        default_norm_params=None,
        default_activation="relu",
        default_dropout=0,
        default_pool_function="max",
        default_pool_size=None,
        default_pool_strides=None,
        default_pool_padding="valid",
        **kwargs,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.in_channels = in_channels

        if stacked_layers is None:
            self.stacked_parallel_layers = [
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
            ]

        else:
            self.stacked_parallel_layers = stacked_layers

        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            for j in range(len(parallel_layers)):
                layer = parallel_layers[j]
                if "num_filters" not in layer:
                    layer["num_filters"] = default_num_filters
                if "filter_size" not in layer:
                    layer["filter_size"] = default_filter_size
                if "strides" not in layer:
                    layer["strides"] = default_strides
                if "padding" not in layer:
                    layer["padding"] = default_padding
                if "dilation_rate" not in layer:
                    layer["dilation_rate"] = default_dilation_rate
                if "use_bias" not in layer:
                    layer["use_bias"] = default_use_bias
                if "weights_initializer" not in layer:
                    layer["weights_initializer"] = default_weights_initializer
                if "bias_initializer" not in layer:
                    layer["bias_initializer"] = default_bias_initializer
                if "norm" not in layer:
                    layer["norm"] = default_norm
                if "norm_params" not in layer:
                    layer["norm_params"] = default_norm_params
                if "activation" not in layer:
                    layer["activation"] = default_activation
                if "dropout" not in layer:
                    layer["dropout"] = default_dropout
                if "pool_function" not in layer:
                    layer["pool_function"] = default_pool_function
                if "pool_size" not in layer:
                    if i == len(self.stacked_parallel_layers) - 1:
                        layer["pool_size"] = default_pool_size
                    else:
                        layer["pool_size"] = None
                if "pool_strides" not in layer:
                    layer["pool_strides"] = default_pool_strides
                if "pool_padding" not in layer:
                    layer["pool_padding"] = default_pool_padding

        self.stack = nn.ModuleList()
        num_channels = self.in_channels
        sequence_length = self.max_sequence_length
        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            logger.debug(f"   stack layer {i}")
            self.stack.append(ParallelConv1D(num_channels, sequence_length, layers=parallel_layers))

            logger.debug(
                f"{self.__class__.__name__} layer {i}, input shape "
                f"{self.stack[i].input_shape}, output shape "
                f"{self.stack[i].output_shape}"
            )

            # set input specification for the layer
            num_channels = self.stack[i].output_shape[1]
            sequence_length = self.stack[i].output_shape[0]

    @property
    def input_shape(self):
        """Returns the size of the input tensor without the batch dimension."""
        return torch.Size([self.max_sequence_length, self.in_channels])

    def forward(self, inputs, mask=None):
        hidden = inputs

        for layer in self.stack:
            hidden = layer(hidden)

        if hidden.shape[2] == 0:
            raise ValueError(
                "The output of the conv stack has the second dimension "
                "(length of the sequence) equal to 0. "
                "This means that the combination of filter_size, padding, "
                "stride, pool_size, pool_padding and pool_stride is reduces "
                "the sequence length more than is possible. "
                'Try using "same" padding and reducing or eliminating stride '
                "and pool."
            )

        return hidden


class Conv2DLayer(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        out_channels: int = 256,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int], str] = "valid",
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "zeros",
        norm: Optional[str] = None,
        norm_params: Optional[Dict[str, Any]] = None,
        activation: str = "relu",
        dropout: float = 0,
        pool_function: int = "max",
        pool_kernel_size: Union[int, Tuple[int]] = None,
        pool_stride: Optional[int] = None,
        pool_padding: Union[int, Tuple[int]] = 0,
        pool_dilation: Union[int, Tuple[int]] = 1,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        self._input_shape = (in_channels, img_height, img_width)
        pool_stride = pool_stride or pool_kernel_size

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                padding_mode=padding_mode,
            )
        )
        out_height, out_width = get_img_output_shape(img_height, img_width, kernel_size, stride, padding, dilation)

        if norm and norm_params is None:
            norm_params = {}
        if norm == "batch":
            # Batch norm over channels
            self.layers.append(nn.BatchNorm2d(num_features=out_channels, **norm_params))
        elif norm == "layer":
            # Layer norm over image height and width
            self.layers.append(nn.LayerNorm(normalized_shape=(out_height, out_width), **norm_params))

        self.layers.append(get_activation(activation))

        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        if pool_kernel_size is not None:
            pool = partial(nn.MaxPool2d, dilation=pool_dilation)
            if pool_function in {"average", "avg", "mean"}:
                pool = nn.AvgPool2d
            self.layers.append(pool(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
            out_height, out_width = get_img_output_shape(
                img_height=out_height,
                img_width=out_width,
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
                dilation=pool_dilation,
            )

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = (out_channels, out_height, out_width)

    def forward(self, inputs):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden)

        return hidden

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


class Conv2DStack(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        layers: Optional[List[Dict]] = None,
        num_layers: Optional[int] = None,
        first_in_channels: Optional[int] = None,
        default_out_channels: int = 256,
        default_kernel_size: Union[int, Tuple[int]] = 3,
        default_stride: Union[int, Tuple[int]] = 1,
        default_padding: Union[int, Tuple[int], str] = "valid",
        default_dilation: Union[int, Tuple[int]] = 1,
        default_groups: int = 1,
        default_use_bias: bool = True,
        default_padding_mode: str = "zeros",
        default_norm: Optional[str] = None,
        default_norm_params: Optional[Dict[str, Any]] = None,
        default_activation: str = "relu",
        default_dropout: int = 0,
        default_pool_function: int = "max",
        default_pool_kernel_size: Union[int, Tuple[int]] = 2,
        default_pool_stride: Union[int, Tuple[int]] = None,
        default_pool_padding: Union[int, Tuple[int]] = 0,
        default_pool_dilation: Union[int, Tuple[int]] = 1,
    ):
        super().__init__()

        # Confirm that all inputs are consistent
        first_in_channels = self._check_in_channels(first_in_channels, layers)
        default_pool_stride = default_pool_stride or default_pool_kernel_size
        if layers is not None and num_layers is not None:
            raise Warning("Both layers and num_layers are not None." "Default to using layers.")
        if (
            first_in_channels is not None
            and layers is not None
            and len(layers) > 0
            and "in_channels" in layers[0]
            and layers[0]["in_channels"] != first_in_channels
        ):
            raise Warning(
                "Input channels is set via layers[0]['in_channels'] and first_in_channels."
                "Default to using first_in_channels."
            )

        self._input_shape = (first_in_channels, img_height, img_width)

        if layers is None:
            if num_layers is None:
                self.layers = [
                    {"out_channels": 32},
                    {"out_channels": 64},
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append(
                        {
                            "kernel_size": default_kernel_size,
                            "out_channels": default_out_channels,
                            "pool_kernel_size": default_pool_kernel_size,
                        }
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if "out_channels" not in layer:
                layer["out_channels"] = default_out_channels
            if "kernel_size" not in layer:
                layer["kernel_size"] = default_kernel_size
            if "stride" not in layer:
                layer["stride"] = default_stride
            if "padding" not in layer:
                layer["padding"] = default_padding
            if "dilation" not in layer:
                layer["dilation"] = default_dilation
            if "groups" not in layer:
                layer["groups"] = default_groups
            if "use_bias" not in layer:
                layer["use_bias"] = default_use_bias
            if "padding_mode" not in layer:
                layer["padding_mode"] = default_padding_mode
            if "norm" not in layer:
                layer["norm"] = default_norm
            if "norm_params" not in layer:
                layer["norm_params"] = default_norm_params
            if "activation" not in layer:
                layer["activation"] = default_activation
            if "dropout" not in layer:
                layer["dropout"] = default_dropout
            if "pool_function" not in layer:
                layer["pool_function"] = default_pool_function
            if "pool_kernel_size" not in layer:
                layer["pool_kernel_size"] = default_pool_kernel_size
            if "pool_stride" not in layer:
                layer["pool_stride"] = default_pool_stride
            if "pool_padding" not in layer:
                layer["pool_padding"] = default_pool_padding
            if "pool_dilation" not in layer:
                layer["pool_dilation"] = default_pool_dilation

        self.stack = torch.nn.ModuleList()

        in_channels = first_in_channels
        for i, layer in enumerate(self.layers):
            logger.debug(f"   stack layer {i}")
            self.stack.append(
                Conv2DLayer(
                    img_height=img_height,
                    img_width=img_width,
                    in_channels=in_channels,
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    dilation=layer["dilation"],
                    groups=layer["groups"],
                    use_bias=layer["use_bias"],
                    padding_mode=layer["padding_mode"],
                    norm=layer["norm"],
                    norm_params=layer["norm_params"],
                    activation=layer["activation"],
                    dropout=layer["dropout"],
                    pool_function=layer["pool_function"],
                    pool_kernel_size=layer["pool_kernel_size"],
                    pool_stride=layer["pool_stride"],
                    pool_padding=layer["pool_padding"],
                    pool_dilation=layer["pool_dilation"],
                )
            )
            in_channels, img_height, img_width = self.stack[-1].output_shape

        self._output_shape = (in_channels, img_height, img_width)

    def forward(self, inputs):
        hidden = inputs

        for layer in self.stack:
            hidden = layer(hidden)

        return hidden

    def _check_in_channels(self, first_in_channels: Optional[int], layers: Optional[List[Dict]]) -> None:
        """Confirms that in_channels for first layer of the stack exists."""

        if first_in_channels is not None:
            return first_in_channels
        elif layers is not None and len(layers) > 0 and "in_channels" in layers[0]:
            return layers[0]["in_channels"]
        raise ValueError(
            "In_channels for first layer should be specified either via " "`first_in_channels` or `layers` arguments."
        )

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return torch.size(self._input_shape)


class Conv2DLayerFixedPadding(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        out_channels=256,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self._input_shape = (in_channels, img_height, img_width)

        padding = "same"
        if stride > 1:
            padding = (kernel_size - 1) // 2

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
            )
        )
        img_height, img_width = get_img_output_shape(
            img_height=img_height,
            img_width=img_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = (out_channels, img_height, img_width)

    def forward(self, inputs):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden)

        return hidden

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)


class ResNetBlock(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        stride: int = 1,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 0.001,
        projection_shortcut: Optional[LudwigModule] = None,
    ):
        """Resnet blocks used for ResNet34 and smaller.

        stride: A single int specifying the stride of the first convolution.
            The last convolution will have stride of 1.
        """
        super().__init__()
        self._input_shape = (first_in_channels, img_height, img_width)

        self.conv1 = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=first_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        )
        in_channels, img_height, img_width = self.conv1.output_shape
        self.norm1 = nn.BatchNorm2d(num_features=in_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.relu1 = get_activation("relu")

        self.conv2 = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )
        self.norm2 = nn.BatchNorm2d(num_features=out_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.relu2 = get_activation("relu")

        for layer in [self.conv1, self.norm1, self.relu1, self.conv2, self.norm2, self.relu2]:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = self.conv2.output_shape

        self.projection_shortcut = projection_shortcut
        if self.projection_shortcut is not None and self.projection_shortcut.output_shape != self._output_shape:
            raise ValueError(
                f"Output shapes of ResnetBlock and projection_shortcut should "
                f"match but are {self._output_shape} and "
                f"{self.projection_shortcut.output_shape} respectively."
            )
        if self.projection_shortcut is None and self._input_shape != self._output_shape:
            self.projection_shortcut = Conv2DLayer(
                img_height=self._input_shape[1],
                img_width=self._input_shape[2],
                in_channels=first_in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )

    def forward(self, inputs):
        shortcut = inputs

        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(shortcut)

        hidden = self.conv1(inputs)
        hidden = self.norm1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)

        return self.relu2(hidden + shortcut)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)


# TODO(shreya): Combine with ResNetBlock by adding a flag.
class ResNetBottleneckBlock(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        stride: int = 1,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 0.001,
        projection_shortcut: Optional[LudwigModule] = None,
    ):
        """Resnet bottleneck blocks used for ResNet50 and larger.

        stride: A single int specifying the stride of the middle convolution.
            The first and last convolution will have stride of 1.
        """
        super().__init__()

        self._input_shape = (first_in_channels, img_height, img_width)

        self.conv1 = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=first_in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        in_channels, img_height, img_width = self.conv1.output_shape
        self.norm1 = nn.BatchNorm2d(num_features=in_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.relu1 = get_activation("relu")

        self.conv2 = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        )
        in_channels, img_height, img_width = self.conv2.output_shape
        self.norm2 = nn.BatchNorm2d(num_features=in_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.relu2 = get_activation("relu")

        self.conv3 = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=in_channels,
            out_channels=4 * out_channels,
            kernel_size=1,
            stride=1,
        )
        self.norm3 = nn.BatchNorm2d(num_features=4 * out_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.relu3 = get_activation("relu")

        for layer in [
            self.conv1,
            self.norm1,
            self.relu1,
            self.conv2,
            self.norm2,
            self.relu2,
            self.conv3,
            self.norm3,
            self.relu3,
        ]:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = self.conv3.output_shape

        self.projection_shortcut = projection_shortcut
        if self.projection_shortcut is not None and self.projection_shortcut.output_shape != self._output_shape:
            raise ValueError(
                f"Output shapes of ResnetBlock and projection_shortcut should "
                f"match but are {self._output_shape} and "
                f"{self.projection_shortcut.output_shape} respectively."
            )
        if self.projection_shortcut is None and self._input_shape != self._output_shape:
            self.projection_shortcut = Conv2DLayer(
                img_height=self._input_shape[1],
                img_width=self._input_shape[2],
                in_channels=first_in_channels,
                out_channels=4 * out_channels,
                kernel_size=1,
                stride=stride,
            )

    def forward(self, inputs):
        shortcut = inputs

        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(shortcut)

        hidden = self.conv1(inputs)
        hidden = self.norm1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.conv3(hidden)
        hidden = self.norm3(hidden)

        return self.relu3(hidden + shortcut)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


class ResNetBlockLayer(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        is_bottleneck: bool,
        block_fn: Union[ResNetBlock, ResNetBottleneckBlock],
        num_blocks: int,
        stride: Union[int, Tuple[int]] = 1,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 0.001,
    ):
        super().__init__()

        self._input_shape = (first_in_channels, img_height, img_width)

        # Bottleneck blocks end with 4x the number of channels as they start with
        projection_out_channels = out_channels * 4 if is_bottleneck else out_channels
        projection_shortcut = Conv2DLayerFixedPadding(
            img_height=img_height,
            img_width=img_width,
            in_channels=first_in_channels,
            out_channels=projection_out_channels,
            kernel_size=1,
            stride=stride,
        )

        self.layers = torch.nn.ModuleList(
            [
                block_fn(
                    img_height,
                    img_width,
                    first_in_channels,
                    out_channels,
                    stride,
                    batch_norm_momentum,
                    batch_norm_epsilon,
                    projection_shortcut,
                )
            ]
        )
        in_channels, img_height, img_width = self.layers[-1].output_shape

        for _ in range(1, num_blocks):
            self.layers.append(
                block_fn(
                    img_height=img_height,
                    img_width=img_width,
                    first_in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_epsilon=batch_norm_epsilon,
                )
            )
            in_channels, img_height, img_width = self.layers[-1].output_shape

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = (in_channels, img_height, img_width)

    def forward(self, inputs):
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


class ResNet(LudwigModule):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        first_in_channels: int,
        out_channels: int,
        resnet_size: int = 34,
        kernel_size: Union[int, Tuple[int]] = 7,
        conv_stride: Union[int, Tuple[int]] = 2,
        first_pool_kernel_size: Union[int, Tuple[int]] = 3,
        first_pool_stride: Union[int, Tuple[int]] = 2,
        block_sizes: List[int] = None,
        block_strides: List[Union[int, Tuple[int]]] = None,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 0.001,
    ):
        """Creates a model obtaining an image representation.

        Implements ResNet v2:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
          resnet_size: A single integer for the size of the ResNet model.
          is_bottleneck: Use regular blocks or bottleneck blocks.
          out_channels: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_kernel_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_kernel_size is None.
          block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
          block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
        Raises:
          ValueError: if invalid version is selected.
        """
        super().__init__()

        self._input_shape = (first_in_channels, img_height, img_width)

        is_bottleneck = self.get_is_bottleneck(resnet_size, block_sizes)
        block_class = self.get_block_fn(is_bottleneck)
        block_sizes, block_strides = self.get_blocks(resnet_size, block_sizes, block_strides)

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            Conv2DLayerFixedPadding(
                img_height=img_height,
                img_width=img_width,
                in_channels=first_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
            )
        )
        in_channels, img_height, img_width = self.layers[-1].output_shape
        self.layers.append(
            nn.BatchNorm2d(num_features=out_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        )
        self.layers.append(get_activation("relu"))

        if first_pool_kernel_size:
            self.layers.append(nn.MaxPool2d(kernel_size=first_pool_kernel_size, stride=first_pool_stride, padding=1))
            img_height, img_width = get_img_output_shape(
                img_height=img_height,
                img_width=img_width,
                kernel_size=first_pool_kernel_size,
                stride=first_pool_stride,
                padding=1,
                dilation=1,
            )

        for i, num_blocks in enumerate(block_sizes):
            self.layers.append(
                ResNetBlockLayer(
                    img_height=img_height,
                    img_width=img_width,
                    first_in_channels=in_channels,
                    out_channels=out_channels,
                    is_bottleneck=is_bottleneck,
                    block_fn=block_class,
                    num_blocks=num_blocks,
                    stride=block_strides[i],
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_epsilon=batch_norm_epsilon,
                )
            )
            out_channels *= 2
            in_channels, img_height, img_width = self.layers[-1].output_shape

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

        self._output_shape = (in_channels, img_height, img_width)

    def get_is_bottleneck(self, resnet_size: int, block_sizes: List[int]) -> bool:
        if (resnet_size is not None and resnet_size >= 50) or (block_sizes is not None and sum(block_sizes) >= 16):
            return True
        return False

    def get_block_fn(self, is_bottleneck: bool) -> Union[ResNetBlock, ResNetBottleneckBlock]:
        if is_bottleneck:
            return ResNetBottleneckBlock
        return ResNetBlock

    def get_blocks(self, resnet_size: int, block_sizes: List[int], block_strides: List[int]) -> Tuple[List[int]]:
        if block_sizes is None:
            block_sizes = get_resnet_block_sizes(resnet_size)
        if block_strides is None:
            block_strides = [1] + [2 for _ in range(len(block_sizes) - 1)]
        return block_sizes, block_strides

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)

        return hidden

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


################################################################################
# The following code for ResNet is adapted from the TensorFlow implementation
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
################################################################################

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
resnet_choices = {
    8: [1, 2, 2],
    14: [1, 2, 2],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
}


def get_resnet_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
    Returns:
      A list of block sizes to use in building the model.
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    try:
        return resnet_choices[resnet_size]
    except KeyError:
        err = "Could not find layers for selected Resnet size.\n" "Size received: {}; sizes allowed: {}.".format(
            resnet_size, resnet_choices.keys()
        )
        raise ValueError(err)
