from dataclasses import Field
from typing import Optional
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata


def DropoutField(description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or (
        "Default dropout rate applied to fully connected layers. "
        "Increasing dropout is a common form of regularization to combat overfitting."
    )
    return schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description=description,
        parameter_metadata=parameter_metadata,
    )


def NumFCLayersField(default: int = 0, description: str = None, parameter_metadata: ParameterMetadata = None) -> Field:
    description = description or (
        "Number of stacked fully connected layers to apply. "
        "Increasing layers adds capacity to the model, enabling it to learn more complex feature interactions."
    )
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
    return schema_utils.StringOptions(
        ["batch", "layer", "ghost"],
        default=default,
        allow_none=True,
        description="",
        parameter_metadata=parameter_metadata,
    )
