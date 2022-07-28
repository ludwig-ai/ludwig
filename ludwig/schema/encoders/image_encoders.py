from typing import Any, Dict, List, Optional, Tuple, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.utils.torch_utils import initializer_registry


@dataclass
class Stacked2DCNNEncoderConfig(BaseEncoderConfig):

    type: str = "stacked_cnn"

    conv_layers: Optional[List[dict]] = schema_utils.DictList(
        default=None,
        description="List of convolutional layers to use in the encoder. ",
    )

    num_conv_layers: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of convolutional layers to use in the encoder. ",
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of channels to use in the encoder. ",
    )

    out_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
    )

    kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=3,
        description="An integer or pair of integers specifying the kernel size. A single integer specifies a square "
        "kernel, while a pair of integers specifies the height and width of the kernel in that order (h, "
        "w). If a kernel_size is not specified in conv_layers this kernel_size that will be used for "
        "each layer.",
    )

    stride: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=1,
        description="An integer or pair of integers specifying the stride of the convolution along the height and "
        "width. If a stride is not already specified in conv_layers, specifies the default stride of the "
        "2D convolutional kernel that will be used for each layer. ",
    )

    padding: Optional[Union[int, Tuple[int], str]] = schema_utils.PositiveIntegerOrTupleOrStringOptions(
        options=["valid", "same"],
        default="valid",
        description="An int, pair of ints (h, w), or one of valid, same specifying the padding used for convolution "
        "kernels. ",
    )

    dilation: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=1,
        description="An int or pair of ints specifying the dilation rate to use for dilated convolution. If dilation "
        "is not already specified in conv_layers, specifies the default dilation of the 2D convolutional "
        "kernel that will be used for each layer.",
    )

    groups: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="Groups controls the connectivity between convolution inputs and outputs. When groups = 1, each "
        "output channel depends on every input channel. When groups > 1, input and output channels are "
        "divided into groups separate groups, where each output channel depends only on the inputs in its "
        "respective input channel group. in_channels and out_channels must both be divisible by groups.",
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
    )

    conv_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="If a norm is not already specified in conv_layers this is the default norm that will be used for "
        "each layer. It indicates the normalization applied to the activations and can be null, "
        "batch or layer.",
    )

    conv_norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="Parameters used if conv_norm is either batch or layer. ",
    )

    conv_activation: str = schema_utils.ActivationOptions(
        description="If an activation is not already specified in conv_layers this is the default activation that "
        "will be used for each layer. It indicates the activation function applied to the output.",
    )

    conv_dropout: Optional[int] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
    )

    pool_function: Optional[str] = schema_utils.StringOptions(
        ["max", "average", "avg", "mean"],
        default="max",
        description="Pooling function to use.",
    )

    pool_kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=2,
        description="An integer or pair of integers specifying the pooling size. If pool_kernel_size is not specified "
        "in conv_layers this is the default value that will be used for each layer.",
    )

    pool_stride: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=None,
        description="An integer or pair of integers specifying the pooling stride, which is the factor by which the "
        "pooling layer downsamples the feature map. Defaults to pool_kernel_size.",
    )

    pool_padding: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=0,
        description="An integer or pair of ints specifying pooling padding (h, w).",
    )

    pool_dilation: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=1,
        description="An integer or pair of ints specifying pooling dilation rate (h, w).",
    )

    fc_layers: Optional[Optional[List[Dict]]] = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
    )

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="If output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
    )

    fc_use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    fc_weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
    )

    fc_bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="Initializer for the bias vector.",
    )

    fc_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="If a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
    )

    fc_norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
    )

    fc_activation: Optional[str] = schema_utils.ActivationOptions(
        description="If an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
    )

    fc_dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
    )


@dataclass
class ResNetEncoderConfig(BaseEncoderConfig):

    type: str = "resnet"

    resnet_size: Optional[int] = schema_utils.PositiveInteger(
        default=50,
        description="The size of the ResNet model to use.",
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        description="Number of channels to use in the encoder. ",
    )

    out_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
    )

    kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=3,
        description="An integer or pair of integers specifying the kernel size. A single integer specifies a square "
        "kernel, while a pair of integers specifies the height and width of the kernel in that order (h, "
        "w). If a kernel_size is not specified in conv_layers this kernel_size that will be used for "
        "each layer.",
    )

    conv_stride: Union[int, Tuple[int]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=1,
        description="An integer or pair of integers specifying the stride of the initial convolutional layer.",
    )

    first_pool_kernel_size: Union[int, Tuple[int]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=None,
        description="Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.",
    )

    first_pool_stride: Union[int, Tuple[int]] = schema_utils.IntegerOrSequenceOfIntegers(
        default=None,
        description="Stride for first pooling layer. If null, defaults to first_pool_kernel_size.",
    )

    batch_norm_momentum: float = schema_utils.NonNegativeFloat(
        default=0.9,
        description="Momentum of the batch norm running statistics.",
    )

    batch_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=0.001,
        description="Epsilon of the batch norm.",
    )

    fc_layers: Optional[Optional[List[Dict]]] = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
    )

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="if output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
    )

    use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
    )

    bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="initializer for the bias vector.",
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="if a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
    )

    norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
    )

    activation: Optional[str] = schema_utils.ActivationOptions(
        description="if an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
    )


@dataclass
class MLPMixerEncoderConfig(BaseEncoderConfig):

    type: str = "mlp_mixer"

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
    )

    embed_size: int = schema_utils.PositiveInteger(
        default=512,
        description="The patch embedding size, the output size of the mixer if avg_pool is true.",
    )

    token_size: int = schema_utils.PositiveInteger(
        default=2048,
        description="The per-patch embedding size.",
    )

    channel_dim: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of channels in hidden layer.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="The depth of the network (the number of Mixer blocks).",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )

    avg_pool: bool = schema_utils.Boolean(
        default=True,
        description="If true, pools output over patch dimension, outputs a vector of shape (embed_size). If false, "
        "the output tensor is of shape (n_patches, embed_size), where n_patches is img_height x img_width "
        "/ patch_size².",
    )


@dataclass
class ViTEncoderConfig(BaseEncoderConfig):

    type: str = "vit"

    use_pretrained: bool = (
        schema_utils.Boolean(
            default=True,
            description="Use pre-trained model weights from Hugging Face.",
        ),
    )

    pretrained_model: str = (
        schema_utils.String(  # Internal Only
            default="google/vit-base-patch16-224",
            description="The name of the pre-trained model to use.",
        ),
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooling layer.",
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads in each attention layer.",
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the intermediate (i.e., feed-forward) layer in the Transformer encoder.",
    )

    hidden_act: str = schema_utils.StringOptions(
        ["relu", "gelu", "selu", "gelu_new"],
        default="gelu",
        description="Hidden layer activation, one of gelu, relu, selu or gelu_new.",
    )

    hidden_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for all fully connected layers in the embeddings, encoder, and pooling.",
    )

    attention_probs_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for the attention probabilities.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="",
    )

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
    )
