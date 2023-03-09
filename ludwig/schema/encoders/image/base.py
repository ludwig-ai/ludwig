from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass
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
@ludwig_dataclass
class Stacked2DCNNConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "Stacked2DCNN"

    type: str = schema_utils.ProtectedString(
        "stacked_cnn",
        description=ENCODER_METADATA["Stacked2DCNN"]["type"].long_description,
    )

    conv_dropout: Optional[int] = schema_utils.FloatRange(
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

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
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

    padding_mode: Optional[str] = schema_utils.StringOptions(
        options=["zeros", "reflect", "replicate", "circular"],
        default="zeros",
        description="If padding_mode is not already specified in conv_layers, specifies the default padding_mode of "
        "the 2D convolutional kernel that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["padding_mode"],
    )

    padding: Optional[Union[int, Tuple[int], str]] = schema_utils.OneOfOptionsField(
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

    dilation: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
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

    groups: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="Groups controls the connectivity between convolution inputs and outputs. When groups = 1, each "
        "output channel depends on every input channel. When groups > 1, input and output channels are "
        "divided into groups separate groups, where each output channel depends only on the inputs in its "
        "respective input channel group. in_channels and out_channels must both be divisible by groups.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["groups"],
    )

    pool_function: Optional[str] = schema_utils.StringOptions(
        ["max", "average", "avg", "mean"],
        default="max",
        description="Pooling function to use.",
        parameter_metadata=ENCODER_METADATA["conv_params"]["pool_function"],
    )

    pool_kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
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

    pool_stride: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
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

    pool_padding: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=0,
        allow_none=True,
        description="An integer or pair of ints specifying pooling padding (h, w).",
        field_options=[
            schema_utils.NonNegativeInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_padding"],
    )

    pool_dilation: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        description="An integer or pair of ints specifying pooling dilation rate (h, w).",
        field_options=[
            schema_utils.PositiveInteger(default=None, allow_none=True, description=""),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["pool_dilation"],
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="If output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["output_size"],
    )

    conv_use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="If bias not already specified in conv_layers, specifies if the 2D convolutional kernel should "
        "have a bias term.",
    )

    conv_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
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

    num_conv_layers: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_conv_layers"],
    )

    conv_layers: Optional[List[dict]] = schema_utils.DictList(
        default=None,
        description="List of convolutional layers to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["conv_params"]["conv_layers"],
    )

    fc_dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_dropout"],
    )

    fc_activation: Optional[str] = schema_utils.ActivationOptions(
        description="If an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_activation"],
    )

    fc_use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_use_bias"],
    )

    fc_bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_bias_initializer"],
    )

    fc_weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["fc_weights_initializer"],
    )

    fc_norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
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

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["Stacked2DCNN"]["num_fc_layers"],
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


@DeveloperAPI
@register_encoder_config("_resnet_legacy", IMAGE)
@ludwig_dataclass
class ResNetConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "ResNet"

    type: str = schema_utils.ProtectedString(
        "_resnet_legacy",
        description=ENCODER_METADATA["ResNet"]["type"].long_description,
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate",
        parameter_metadata=ENCODER_METADATA["ResNet"]["dropout"],
    )

    activation: Optional[str] = schema_utils.ActivationOptions(
        description="if an activation is not already specified in fc_layers this is the default activation that will "
        "be used for each layer. It indicates the activation function applied to the output.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["activation"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["width"],
    )

    resnet_size: Optional[int] = schema_utils.PositiveInteger(
        default=50,
        description="The size of the ResNet model to use.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["resnet_size"],
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels to use in the encoder. ",
        parameter_metadata=ENCODER_METADATA["ResNet"]["num_channels"],
    )

    out_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=32,
        description="Indicates the number of filters, and by consequence the output channels of the 2d convolution. "
        "If out_channels is not already specified in conv_layers this is the default out_channels that "
        "will be used for each layer. ",
        parameter_metadata=ENCODER_METADATA["ResNet"]["out_channels"],
    )

    kernel_size: Optional[Union[int, Tuple[int]]] = schema_utils.OneOfOptionsField(
        default=3,
        allow_none=True,
        description="An integer or pair of integers specifying the kernel size. A single integer specifies a square "
        "kernel, while a pair of integers specifies the height and width of the kernel in that order (h, "
        "w). If a kernel_size is not specified in conv_layers this kernel_size that will be used for "
        "each layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNet"]["kernel_size"],
    )

    conv_stride: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        description="An integer or pair of integers specifying the stride of the initial convolutional layer.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNet"]["conv_stride"],
    )

    first_pool_kernel_size: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        description="Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNet"]["first_pool_kernel_size"],
    )

    first_pool_stride: Union[int, Tuple[int]] = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        description="Stride for first pooling layer. If null, defaults to first_pool_kernel_size.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["ResNet"]["first_pool_stride"],
    )

    batch_norm_momentum: float = schema_utils.NonNegativeFloat(
        default=0.9,
        description="Momentum of the batch norm running statistics.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["batch_norm_momentum"],
    )

    batch_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=0.001,
        description="Epsilon of the batch norm.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["batch_norm_epsilon"],
    )

    use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["use_bias"],
    )

    bias_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="zeros",
        description="initializer for the bias vector.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["bias_initializer"],
    )

    weights_initializer: Optional[str] = schema_utils.StringOptions(
        sorted(list(initializer_registry.keys())),
        default="xavier_uniform",
        description="Initializer for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["weights_initializer"],
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="if output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer. ",
        parameter_metadata=ENCODER_METADATA["ResNet"]["output_size"],
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="if a norm is not already specified in fc_layers this is the default norm that will be used for "
        "each layer. It indicates the norm of the output and can be null, batch or layer.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["norm"],
    )

    norm_params: Optional[Dict[str, Any]] = schema_utils.Dict(
        default=None,
        description="parameters used if norm is either batch or layer. For information on parameters used with batch "
        "see Torch's documentation on batch normalization or for layer see Torch's documentation on layer "
        "normalization.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["norm_params"],
    )

    num_fc_layers: Optional[Optional[int]] = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["ResNet"]["num_fc_layers"],
    )

    fc_layers: Optional[Optional[List[Dict]]] = schema_utils.DictList(
        default=None,
        description="A list of dictionaries containing the parameters of all the fully connected layers. The length "
        "of the list determines the number of stacked fully connected layers and the content of each "
        "dictionary determines the parameters for a specific layer. The available parameters for each "
        "layer are: activation, dropout, norm, norm_params, output_size, use_bias, bias_initializer and "
        "weights_initializer. If any of those values is missing from the dictionary, the default one "
        "specified as a parameter of the encoder will be used instead. ",
        parameter_metadata=ENCODER_METADATA["ResNet"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("mlp_mixer", IMAGE)
@ludwig_dataclass
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
@register_encoder_config("_vit_legacy", IMAGE)
@ludwig_dataclass
class ViTConfig(ImageEncoderConfig):
    @staticmethod
    def module_name():
        return "ViT"

    type: str = schema_utils.ProtectedString(
        "_vit_legacy",
        description=ENCODER_METADATA["ViT"]["type"].long_description,
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the input image.",
        parameter_metadata=ENCODER_METADATA["ViT"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the input image.",
        parameter_metadata=ENCODER_METADATA["ViT"]["width"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ViT"]["num_hidden_layers"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooling layer.",
        parameter_metadata=ENCODER_METADATA["ViT"]["hidden_size"],
    )

    hidden_act: str = schema_utils.StringOptions(
        ["relu", "gelu", "selu", "gelu_new"],
        default="gelu",
        description="Hidden layer activation, one of gelu, relu, selu or gelu_new.",
        parameter_metadata=ENCODER_METADATA["ViT"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for all fully connected layers in the embeddings, encoder, and pooling.",
        parameter_metadata=ENCODER_METADATA["ViT"]["hidden_dropout_prob"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads in each attention layer.",
        parameter_metadata=ENCODER_METADATA["ViT"]["num_attention_heads"],
    )

    attention_probs_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout rate for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ViT"]["attention_probs_dropout_prob"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the intermediate (i.e., feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ViT"]["intermediate_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ViT"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ViT"]["layer_norm_eps"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="",
        parameter_metadata=ENCODER_METADATA["ViT"]["gradient_checkpointing"],
    )

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="The image patch size. Each patch is patch_size² pixels. Must evenly divide the image width and "
        "height.",
        parameter_metadata=ENCODER_METADATA["ViT"]["patch_size"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ViT"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
        parameter_metadata=ENCODER_METADATA["ViT"]["trainable"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Use pre-trained model weights from Hugging Face.",
        parameter_metadata=ENCODER_METADATA["ViT"]["use_pretrained"],
    )

    pretrained_model: str = schema_utils.String(
        default="google/vit-base-patch16-224",
        description="The name of the pre-trained model to use.",
        parameter_metadata=ENCODER_METADATA["ViT"]["pretrained_model"],
    )

    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "ImagePreprocessingConfig"):
        """If the encoder is not in trainable mode, override the image width and height to be compatible with the
        pretrained encoder image dimension requirements."""
        if self.requires_equal_dimensions() and self.required_width() != self.required_height():
            raise ValueError("Invalid definition. `required_width` and `required_height` are not equal")

        preprocessing.requires_equal_dimensions = self.requires_equal_dimensions()
        if not self.trainable or self.use_pretrained:
            preprocessing.height = self.required_height()
            preprocessing.width = self.required_width()

    @classmethod
    def requires_equal_dimensions(cls) -> bool:
        return True

    @classmethod
    def required_width(cls) -> Optional[int]:
        return 224

    @classmethod
    def required_height(cls) -> Optional[int]:
        return 224

    def is_pretrained(self) -> bool:
        return self.use_pretrained
