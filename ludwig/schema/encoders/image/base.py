from typing import Any, TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.utils.torch_utils import initializer_registry

if TYPE_CHECKING:
    from ludwig.schema.features.preprocessing.image import ImagePreprocessingConfig


class ImageEncoderConfig(BaseEncoderConfig):
    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "ImagePreprocessingConfig"):
        preprocessing.requires_equal_dimensions = False
        preprocessing.height = None
        preprocessing.width = None


@DeveloperAPI
@register_encoder_config("stacked_cnn", IMAGE)
class Stacked2DCNNConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "Stacked2DCNN"

    type: str = schema_utils.ProtectedString(
        "stacked_cnn",
        description=ENCODER_METADATA["Stacked2DCNN"]["type"].long_description,
    )

    conv_dropout: int | None = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_dropout"],
    )

    conv_activation: str = schema_utils.ActivationOptions(
        description="If an activation is not already specified in conv_layers this is the default activation that "
        "will be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_activation"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["width"],
    )

    num_channels: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_channels"],
    )

    out_channels: int | None = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["out_channels"],
    )

    kernel_size: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=3,
        description="An integer or pair of integers specifying the kernel size. A single integer specifies a square "
        "kernel, while a pair of integers specifies the height and width of the kernel in that order (h, "
        "w). If a kernel_size is not specified in conv_layers this kernel_size that will be used for "
        "each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=3),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["kernel_size"],
    )

    stride: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=1,
        description="An integer or pair of integers specifying the stride of the convolution along the height and "
        "width. If a stride is not already specified in conv_layers, specifies the default stride of the "
        "2D convolutional kernel that will be used for each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=1),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["stride"],
    )

    padding_mode: str | None = schema_utils.StringOptions(
        options=["zeros", "reflect", "replicate", "circular"],
        default="zeros",
        description="If padding_mode is not already specified in conv_layers, specifies the default padding_mode of "
        "the 2D convolutional kernel that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["padding_mode"],
    )

    padding: int | tuple[int] | str | None = schema_utils.OneOfOptionsField(
        default="valid",
        allow_none=True,
        description="An int, pair of ints (h, w), or one of ['valid', 'same'] specifying the padding used for"
        "convolution kernels.",
        field_options=[
            schema_utils.NonNegativeInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
            schema_utils.StringOptions(options=["valid", "same"], default="valid", allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["padding"],
    )

    dilation: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        description="An int or pair of ints specifying the dilation rate to use for dilated convolution. If dilation "
        "is not already specified in conv_layers, specifies the default dilation of the 2D convolutional "
        "kernel that will be used for each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["dilation"],
    )

    groups: int | None = schema_utils.PositiveInteger(
        default=1,
        description="Groups controls the connectivity between convolution inputs and outputs. When groups = 1, each "
        "output channel depends on every input channel. When groups > 1, input and output channels are "
        "divided into groups separate groups, where each output channel depends only on the inputs in its "
        "respective input channel group. in_channels and out_channels must both be divisible by groups.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["groups"],
    )

    pool_function: str | None = schema_utils.StringOptions(
        ["max", "average", "avg", "mean"],
        default="max",
        description="Pooling function to use.",
        parameter_metadata=ENCODER_METADATA["conv_params"]["pool_function"],
    )

    pool_kernel_size: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=2,
        allow_none=True,
        description="An integer or pair of integers specifying the pooling size. If pool_kernel_size is not specified "
        "in conv_layers this is the default value that will be used for each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_kernel_size"],
    )

    pool_stride: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        description="An integer or pair of integers specifying the pooling stride, which is the factor by which the "
        "pooling layer downsamples the feature map. Defaults to pool_kernel_size.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_stride"],
    )

    pool_padding: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=0,
        allow_none=True,
        description="An integer or pair of ints specifying pooling padding (h, w).",
        field_options=[
            schema_utils.NonNegativeInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_padding"],
    )

    pool_dilation: int | tuple[int] | None = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        description="An integer or pair of ints specifying pooling dilation rate (h, w).",
        field_options=[
            schema_utils.PositiveInteger(default=None, allow_none=True, description=""),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_dilation"],
    )

    output_size: int | None = schema_utils.PositiveInteger(
        default=128,
        description="If output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["output_size"],
    )

    conv_use_bias: bool | None = schema_utils.Boolean(
        default=True,
        description="If bias not already specified in conv_layers, specifies if the 2D convolutional kernel should "
        "have a bias term.",
    )

    conv_norm: str | None = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="If a norm is not already specified in conv_layers this is the default norm that will be used for "
        "each layer. It indicates the normalization applied to the activations and can be null, "
        "batch or layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_norm"],
    )

    conv_norm_params: dict[str, Any] | None = schema_utils.Dict(
        default=None,
        description="Parameters used if conv_norm is either batch or layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_norm_params"],
    )

    num_conv_layers: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_conv_layers"],
    )

    conv_layers: list[dict] | None = schema_utils.DictList(
        default=None,
        description="List of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["conv_params"]["conv_layers"],
    )

    fc_dropout: float | None = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_dropout"],
    )

    fc_activation: str | None = schema_utils.ActivationOptions(
        description="If an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_activation"],
    )

    fc_use_bias: bool | None = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_use_bias"],
    )

    fc_bias_initializer: str | None = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_bias_initializer"],
    )

    fc_weights_initializer: str | None = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_weights_initializer"],
    )

    fc_norm: str | None = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="If a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_norm"],
    )

    fc_norm_params: dict[str, Any] | None = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_norm_params"],
    )

    num_fc_layers: int | None | None = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_fc_layers"],
    )

    fc_layers: list[dict] | None | None = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("mlp_mixer", IMAGE)
class MLPMixerConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "MLPMixer"

    type: str = schema_utils.ProtectedString(
        "mlp_mixer",
        description=ENCODER_METADATA["MLPMixer"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["dropout"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["width"],
    )

    num_channels: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["num_channels"],
    )

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["patch_size"],
    )

    embed_size: int = schema_utils.PositiveInteger(
        default=512,
        description="The patch embedding size, the output size of the mixer if avg_pool is true.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["embed_size"],
    )

    token_size: int = schema_utils.PositiveInteger(
        default=2048,
        description="The per-patch embedding size.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["token_size"],
    )

    channel_dim: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of channels in hidden layer.",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["channel_dim"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="The depth of the network (the number of Mixer blocks).",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["num_layers"],
    )

    avg_pool: bool = schema_utils.Boolean(
        default=True,
        description="If true, pools output over patch dimension, outputs a vector of shape (embed_size). If false, "
        "the output tensor is of shape (n_patches, embed_size), where n_patches is img_height x img_width "
        "/ patch_size².",
        parameter_metadata=ENCODER_METADATA["MLPMixer"]["avg_pool"],
    )


@DeveloperAPI
@register_encoder_config("unet", IMAGE)
class UNetEncoderConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "UNetEncoder"

    type: str = schema_utils.ProtectedString(
        "unet",
        description=ENCODER_METADATA["UNetEncoder"]["type"].long_description,
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["UNetEncoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["UNetEncoder"]["width"],
    )

    num_channels: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the input image. ",
        parameter_metadata=ENCODER_METADATA["UNetEncoder"]["num_channels"],
    )

    conv_norm: str | None = schema_utils.StringOptions(
        ["batch"],
        default="batch",
        allow_none=True,
        description="This is the default norm that will be used for each double conv layer." "It can be null or batch.",
        parameter_metadata=ENCODER_METADATA["UNetEncoder"]["conv_norm"],
    )
