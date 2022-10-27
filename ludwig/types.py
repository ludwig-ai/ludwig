"""Public Ludwig types."""
from typing import Any, Dict

# Dictionary of parameters used to configure an input or output feature.
FeatureConfigDict = Dict[str, Any]
# Dictionary representation of the ModelConfig object.
ModelConfigDict = Dict[str, Any]
# Training set metadata, which consists of internal configuration parameters.
TrainingSetMetadataDict = Dict[str, Any]
# Dictionary of preprocessing parameters. May be type-defaults global preprocessing or feature-specific preprocessing.
# https://ludwig.ai/latest/configuration/preprocessing
PreprocessingConfigDict = Dict[str, Any]
