from typing import Any, Dict, List, Optional, Tuple, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata.encoder_metadata import ENCODER_METADATA
from ludwig.utils.torch_utils import initializer_registry


@register_encoder_config("stacked_cnn", IMAGE)
@dataclass
class Stacked2DCNNEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["stacked_cnn"],
        default="stacked_cnn",
        allow_none=False,
        description="Type of encoder.",
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["width"],
    )

    conv_layers: Optional[List[dict]] = schema_utils.DictList(
        default=None,
        description="List of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_layers"],
    )

    num_conv_layers: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_conv_layers"],
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_channels"],
    )

    out_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["out_channels"],
    )

    kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
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

    stride: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
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

    padding: Optional[Union[int, Tuple[int], str]] = schema_utils.OneOfOptionsField(
        default="valid",
        description="An int, pair of ints (h, w), or one of ['valid', 'same'] specifying the padding used for"
        "convolution kernels.",
        field_options=[
            schema_utils.NonNegativeInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
            schema_utils.StringOptions(options=["valid", "same"], default="valid", allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["padding"],
    )

    dilation: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=1,
        description="An int or pair of ints specifying the dilation rate to use for dilated convolution. If dilation "
        "is not already specified in conv_layers, specifies the default dilation of the 2D convolutional "
        "kernel that will be used for each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["dilation"],
    )

    groups: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="Groups controls the connectivity between convolution inputs and outputs. When groups = 1, each "
        "output channel depends on every input channel. When groups > 1, input and output channels are "
        "divided into groups separate groups, where each output channel depends only on the inputs in its "
        "respective input channel group. in_channels and out_channels must both be divisible by groups.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["groups"],
    )

    conv_use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="If bias not already specified in conv_layers, specifies if the 2D convolutional kernel should "
        "have a bias term.",
    )

    padding_mode: Optional[str] = schema_utils.StringOptions(
        options=["zeros", "reflect", "replicate", "circular"],
        default="zeros",
        description="If padding_mode is not already specified in conv_layers, specifies the default padding_mode of "
        "the 2D convolutional kernel that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["padding_mode"],
    )

    conv_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="If a norm is not already specified in conv_layers this is the default norm that will be used for "
        "each layer. It indicates the normalization applied to the activations and can be null, "
        "batch or layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_norm"],
    )

    conv_norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="Parameters used if conv_norm is either batch or layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_norm_params"],
    )

    conv_activation: str = schema_utils.ActivationOptions(
        description="If an activation is not already specified in conv_layers this is the default activation that "
        "will be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_activation"],
    )

    conv_dropout: Optional[int] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["conv_dropout"],
    )

    pool_function: Optional[str] = schema_utils.StringOptions(
        ["max", "average", "avg", "mean"],
        default="max",
        description="Pooling function to use.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_function"],
    )

    pool_kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=2,
        description="An integer or pair of integers specifying the pooling size. If pool_kernel_size is not specified "
        "in conv_layers this is the default value that will be used for each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_kernel_size"],
    )

    pool_stride: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=None,
        description="An integer or pair of integers specifying the pooling stride, which is the factor by which the "
        "pooling layer downsamples the feature map. Defaults to pool_kernel_size.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_stride"],
    )

    pool_padding: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=0,
        description="An integer or pair of ints specifying pooling padding (h, w).",
        field_options=[
            schema_utils.NonNegativeInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_padding"],
    )

    pool_dilation: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=1,
        description="An integer or pair of ints specifying pooling dilation rate (h, w).",
        field_options=[
            schema_utils.PositiveInteger(description="", default=None, allow_none=False),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_dilation"],
    )

    fc_layers: Optional[Optional[List[Dict]]] = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_layers"],
    )

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_fc_layers"],
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="If output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["output_size"],
    )

    fc_use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_use_bias"],
    )

    fc_weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_weights_initializer"],
    )

    fc_bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_bias_initializer"],
    )

    fc_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="If a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_norm"],
    )

    fc_norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_norm_params"],
    )

    fc_activation: Optional[str] = schema_utils.ActivationOptions(
        description="If an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_activation"],
    )

    fc_dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_dropout"],
    )


@register_encoder_config("resnet", IMAGE)
@dataclass
class ResNetEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["resnet"],
        default="resnet",
        allow_none=False,
        description="Type of encoder.",
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["width"],
    )

    resnet_size: Optional[int] = schema_utils.PositiveInteger(
        default=50,
        description="The size of the ResNet model to use.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["resnet_size"],
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["num_channels"],
    )

    out_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["out_channels"],
    )

    kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=3,
        description="An integer or pair of integers specifying the kernel size. A single integer specifies a square "
        "kernel, while a pair of integers specifies the height and width of the kernel in that order (h, "
        "w). If a kernel_size is not specified in conv_layers this kernel_size that will be used for "
        "each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["kernel_size"],
    )

    conv_stride: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=1,
        description="An integer or pair of integers specifying the stride of the initial convolutional layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["conv_stride"],
    )

    first_pool_kernel_size: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=None,
        description="Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["first_pool_kernel_size"],
    )

    first_pool_stride: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=None,
        description="Stride for first pooling layer. If null, defaults to first_pool_kernel_size.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["first_pool_stride"],
    )

    batch_norm_momentum: float = schema_utils.NonNegativeFloat(
        default=0.9,
        description="Momentum of the batch norm running statistics.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["batch_norm_momentum"],
    )

    batch_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=0.001,
        description="Epsilon of the batch norm.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["batch_norm_epsilon"],
    )

    fc_layers: Optional[Optional[List[Dict]]] = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["fc_layers"],
    )

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["num_fc_layers"],
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="if output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["output_size"],
    )

    use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["use_bias"],
    )

    weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["weights_initializer"],
    )

    bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["bias_initializer"],
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="if a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["norm"],
    )

    norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["norm_params"],
    )

    activation: Optional[str] = schema_utils.ActivationOptions(
        description="if an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["activation"],
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["ResNetEncoder"]["dropout"],
    )


@register_encoder_config("mlp_mixer", IMAGE)
@dataclass
class MLPMixerEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["mlp_mixer"],
        default="mlp_mixer",
        allow_none=False,
        description="Type of encoder.",
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["width"],
    )

    num_channels: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["num_channels"],
    )

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["patch_size"],
    )

    embed_size: int = schema_utils.PositiveInteger(
        default=512,
        description="The patch embedding size, the output size of the mixer if avg_pool is true.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["embed_size"],
    )

    token_size: int = schema_utils.PositiveInteger(
        default=2048,
        description="The per-patch embedding size.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["token_size"],
    )

    channel_dim: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of channels in hidden layer.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["channel_dim"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="The depth of the network (the number of Mixer blocks).",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["num_layers"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["dropout"],
    )

    avg_pool: bool = schema_utils.Boolean(
        default=True,
        description="If true, pools output over patch dimension, outputs a vector of shape (embed_size). If false, "
        "the output tensor is of shape (n_patches, embed_size), where n_patches is img_height x img_width "
        "/ patch_size².",
        parameter_metadata=ENCODER_METADATA["MLPMixerEncoder"]["avg_pool"],
    )


@register_encoder_config("vit", IMAGE)
@dataclass
class ViTEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["vit"],
        default="vit",
        allow_none=False,
        description="Type of encoder.",
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["width"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Use pre-trained model weights from Hugging Face.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["use_pretrained"],
    )

    pretrained_model: str = schema_utils.String(
        default="google/vit-base-patch16-224",
        description="The name of the pre-trained model to use.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["pretrained_model"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["saved_weights_in_checkpoint"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooling layer.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads in each attention layer.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the intermediate (i.e., feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["intermediate_size"],
    )

    hidden_act: str = schema_utils.StringOptions(
        ["relu", "gelu", "selu", "gelu_new"],
        default="gelu",
        description="Hidden layer activation, one of gelu, relu, selu or gelu_new.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for all fully connected layers in the embeddings, encoder, and pooling.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["attention_probs_dropout_prob"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["layer_norm_eps"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["gradient_checkpointing"],
    )

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["patch_size"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
        parameter_metadata=ENCODER_METADATA["ViTEncoder"]["trainable"],
    )


@dataclass
class TVBaseEncoderConfig(BaseEncoderConfig):
    use_pretrained: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Download model weights from pre-trained model.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["use_pretrained"],
    )

    model_cache_dir: str = schema_utils.String(
        description="Directory path to cache pretrained model weights.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["model_cache_dir"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["trainable"],
    )


@register_encoder_config("resnet_torch", IMAGE)
@dataclass
class TVResNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.StringOptions(
        ["resnet_torch"],
        default="resnet_torch",
        allow_none=False,
        description="Type of encoder.",
    )

    model_variant: Optional[int] = schema_utils.IntegerOptions(
        [18, 34, 50, 101, 152],
        default=50,
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVResNetEncoder"]["model_variant"],
    )


@register_encoder_config("vgg", IMAGE)
@dataclass
class TVVGGEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.StringOptions(
        ["vgg"],
        default="vgg",
        allow_none=False,
        description="Type of encoder.",
    )

    model_variant: Union[int, str] = schema_utils.OneOfOptionsField(
        default=11,
        description="Pretrained model variant to use.",
        field_options=[
            schema_utils.IntegerOptions(
                [
                    11,
                    13,
                    16,
                    19,
                ],
                default=11,
                allow_none=False,
            ),
            schema_utils.StringOptions(
                [
                    "11_bn",
                    "13_bn",
                    "16_bn",
                    "19_bn",
                ],
                default="11_bn",
                allow_none=False,
            ),
        ],
        parameter_metadata=ENCODER_METADATA["TVVGGEncoder"]["model_variant"],
    )


@register_encoder_config("alexnet", IMAGE)
@dataclass
class TVAlexNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.StringOptions(
        ["alexnet"],
        default="alexnet",
        allow_none=False,
        description="Type of encoder.",
    )

    model_variant: Optional[int] = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVAlexNetEncoder"]["model_variant"],
    )


@register_encoder_config("efficientnet", IMAGE)
@dataclass
class TVEfficientNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.StringOptions(
        ["efficientnet"],
        default="efficientnet",
        allow_none=False,
        description="Type of encoder.",
    )

    model_variant: Optional[int] = schema_utils.StringOptions(
        [
            "b0",
            "b1",
            "b2",
            "b3",
            "b4",
            "b5",
            "b6",
            "b7",
            "v2_",
            "v2_m",
            "v2_l",
        ],
        default="b0",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVEfficientNetEncoder"]["model_variant"],
    )
