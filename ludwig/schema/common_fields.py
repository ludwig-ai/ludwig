from dataclasses import Field
from typing import Optional

from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import COMMON_METADATA
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata
from ludwig.utils.torch_utils import initializer_registry


def DropoutField(default: float = 0.0, description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or (
        "Default dropout rate applied to fully connected layers. "
        "Increasing dropout is a common form of regularization to combat overfitting."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["dropout"]
    return schema_utils.FloatRange(
        default=default,
        min=0,
        max=1,
        description=description,
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
    description = description or (
        "Number of stacked fully connected layers to apply. "
        "Increasing layers adds capacity to the model, enabling it to learn more complex feature interactions."
    )
    parameter_metadata = parameter_metadata or COMMON_METADATA["num_fc_layers"]
    return schema_utils.NonNegativeInteger(
        default=default,
        allow_none=False,
        description=description,
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
    description = description or f"Initializer for the bias vector. Options: {initializers_str}. {INITIALIZER_SUFFIX}"
    parameter_metadata = parameter_metadata or COMMON_METADATA["bias_initializer"]
    return schema_utils.InitializerOrDict(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def WeightsInitializerField(
    default: str = "xavier_uniform", description: str = None, parameter_metadata: ParameterMetadata = None
) -> Field:
    initializers_str = ", ".join([f"`{i}`" for i in initializer_registry.keys()])
    description = description or f"Initializer for the weight matrix. Options: {initializers_str}. {INITIALIZER_SUFFIX}"
    parameter_metadata = parameter_metadata or COMMON_METADATA["weights_initializer"]
    return schema_utils.InitializerOrDict(
        default=default,
        description=description,
        parameter_metadata=parameter_metadata,
    )
