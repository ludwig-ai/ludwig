from typing import List

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class DateEmbedConfig(BaseEncoderConfig):

    type: str = "embed"

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    # TODO (Connor): Add nesting logic for fc_layers, see fully_connected_module.py
    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="The default norm that will be used for each layer.",
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
    )


@dataclass
class DateWaveConfig(BaseEncoderConfig):

    type: str = "wave"

    # TODO (Connor): Add nesting logic for fc_layers, see fully_connected_module.py
    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked fully connected layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="The default norm that will be used for each layer.",
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
    )
