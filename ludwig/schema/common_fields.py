from dataclasses import Field
from typing import Optional

from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import COMMON_METADATA
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata
from ludwig.utils.torch_utils import initializer_registry


def DropoutField(default: float = 0.0, description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or "Default dropout rate applied to fully connected layers."
    full_description = description + (
        " Increasing dropout is a common form of regularization to combat overfitting. "
        "The dropout is expressed as the probability of an element to be zeroed out (0.0 means no dropout)."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["dropout"]
    return schema_utils.FloatRange(
        default=default,
        min=0,
        max=1,
        description=full_description,
        parameter_metadata=parameter_metadata,
    )


def ResidualField(
    default: bool = False, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "Whether to add a residual connection to each fully connected layer block. "
        "Requires all fully connected layers to have the same `output_size`."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["residual"]
    return schema_utils.Boolean(
        default=False,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def NumFCLayersField(default: int = 0, description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or "Number of stacked fully connected layers to apply."
    full_description = description + (
        " Increasing layers adds capacity to the model, enabling it to learn more complex feature interactions."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["num_fc_layers"]
    return schema_utils.NonNegativeInteger(
        default=default,
        allow_none=False,
        description=full_description,
        parameter_metadata=parameter_metadata,
    )


def NormField(
    default: Optional[str] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or "Default normalization applied at the beginnging of fully connected layers."
    parameter_metadata = parameter_metadata or COMMON_METADATA["norm"]
    return schema_utils.StringOptions(
        ["batch", "layer", "ghost"],
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def NormParamsField(description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or "Default parameters passed to the `norm` module."
    parameter_metadata = parameter_metadata or COMMON_METADATA["norm_params"]
    return schema_utils.Dict(
        description=description,
        parameter_metadata=parameter_metadata,
    )


def FCLayersField(description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or (
        "List of dictionaries containing the parameters of all the fully connected layers. "
        "The length of the list determines the number of stacked fully connected layers "
        "and the content of each dictionary determines the parameters for a specific layer. "
        "The available parameters for each layer are: `activation`, `dropout`, `norm`, `norm_params`, "
        "`output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of those values "
        "is missing from the dictionary, the default one provided as a standalone parameter will be used instead."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["fc_layers"]
    return schema_utils.DictList(
        description=description,
        parameter_metadata=parameter_metadata,
    )


INITIALIZER_SUFFIX = """
Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and
other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. For a description of the parameters of each
initializer, see [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html).
"""


def BiasInitializerField(
    default: str = "zeros", description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    initializers_str = ", ".join([f"`{i}`" for i in initializer_registry.keys()])
    description = description or "Initializer for the bias vector."
    full_description = f"{description} Options: {initializers_str}. {INITIALIZER_SUFFIX}"
    parameter_metadata = parameter_metadata or COMMON_METADATA["bias_initializer"]
    return schema_utils.InitializerOrDict(
        default=default,
        description=full_description,
        parameter_metadata=parameter_metadata,
    )


def WeightsInitializerField(
    default: str = "xavier_uniform", description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    initializers_str = ", ".join([f"`{i}`" for i in initializer_registry.keys()])
    description = description or "Initializer for the weight matrix."
    full_description = f"{description} Options: {initializers_str}. {INITIALIZER_SUFFIX}"
    parameter_metadata = parameter_metadata or COMMON_METADATA["weights_initializer"]
    return schema_utils.InitializerOrDict(
        default=default,
        description=full_description,
        parameter_metadata=parameter_metadata,
    )


def EmbeddingInitializerField(
    default: Optional[str] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    initializers_str = ", ".join([f"`{i}`" for i in initializer_registry.keys()])
    description = description or "Initializer for the embedding matrix."
    full_description = f"{description} Options: {initializers_str}."
    parameter_metadata = parameter_metadata or COMMON_METADATA["embedding_initializer"]
    return schema_utils.StringOptions(
        list(initializer_registry.keys()),
        default=default,
        allow_none=True,
        description=full_description,
        parameter_metadata=parameter_metadata,
    )


def EmbeddingSizeField(
    default: int = 256, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for "
        "`dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` "
        "is the number of unique strings appearing in the training set input column plus the number of "
        "special tokens (`<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`)."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["embedding_size"]
    return schema_utils.PositiveInteger(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def EmbeddingsOnCPUField(
    default: bool = False, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "Whether to force the placement of the embedding matrix in regular memory and have the CPU resolve them. "
        "By default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster access, "
        "but in some cases the embedding matrix may be too large. This parameter forces the placement of the "
        "embedding matrix in regular memory and the CPU is used for embedding lookup, slightly slowing down the "
        "process as a result of data transfer between CPU and GPU memory."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["embeddings_on_cpu"]
    return schema_utils.Boolean(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def EmbeddingsTrainableField(
    default: bool = True, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "If `true` embeddings are trained during the training process, if `false` embeddings are fixed. "
        "It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when `representation` is `dense`; `sparse` one-hot encodings are not trainable."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["embeddings_trainable"]
    return schema_utils.Boolean(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def PretrainedEmbeddingsField(
    default: Optional[str] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "Path to a file containing pretrained embeddings. By default `dense` embeddings are initialized "
        "randomly, but this parameter allows to specify a path to a file containing embeddings in the "
        "[GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is "
        "loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. "
        "If the vocabulary contains strings that have no match in the embeddings file, their embeddings are "
        "initialized with the average of all other embedding plus some random noise to make them different "
        "from each other. This parameter has effect only if `representation` is `dense`."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["pretrained_embeddings"]
    return schema_utils.String(
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def MaxSequenceLengthField(
    default: Optional[int] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or "[internal] Maximum sequence length from preprocessing."
    parameter_metadata = parameter_metadata or COMMON_METADATA["max_sequence_length"]
    return schema_utils.PositiveInteger(
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def VocabField(
    default: Optional[list] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or "[internal] Vocabulary for the encoder from preprocessing."
    parameter_metadata = parameter_metadata or COMMON_METADATA["vocab"]
    return schema_utils.List(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def VocabSizeField(
    default: Optional[list] = None, description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or "[internal] Size of the vocabulary from preprocessing."
    parameter_metadata = parameter_metadata or COMMON_METADATA["vocab_size"]
    return schema_utils.PositiveInteger(
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def RepresentationField(
    default: str = "dense", description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "Representation of the embedding. `dense` means the embeddings are initialized randomly, "
        "`sparse` means they are initialized to be one-hot encodings."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["representation"]
    return schema_utils.StringOptions(
        ["dense", "sparse"],
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def ReduceOutputField(
    default: Optional[str] = "sum", description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    description = description or (
        "How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["reduce_output"]
    return schema_utils.ReductionOptions(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )
