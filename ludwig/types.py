"""Public API: Common typing for Ludwig dictionary parameters."""

from typing import Any, Dict

FeatureConfigDict = Dict[str, Any]
"""Dictionary of parameters used to configure an input or output feature.

https://ludwig.ai/latest/configuration/features/supported_data_types/
"""

ModelConfigDict = Dict[str, Any]
"""Dictionary representation of the ModelConfig object.

https://ludwig.ai/latest/configuration/
"""

TrainingSetMetadataDict = Dict[str, Any]
"""Training set metadata, which consists of internal configuration parameters."""

PreprocessingConfigDict = Dict[str, Any]
"""Dictionary of parameters used to configure preprocessing.

May be type-defaults global preprocessing or feature-specific preprocessing.
https://ludwig.ai/latest/configuration/preprocessing/
"""

HyperoptConfigDict = Dict[str, Any]
"""Dictionary of parameters used to configure hyperopt.

https://ludwig.ai/latest/configuration/hyperparameter_optimization/
"""

TrainerConfigDict = Dict[str, Any]
"""Dictionary of parameters used to configure training.

https://ludwig.ai/latest/configuration/trainer/
"""

FeatureTypeDefaultsDict = Dict[str, FeatureConfigDict]
"""Dictionary of type to parameters that configure the defaults for that feature type.

https://ludwig.ai/latest/configuration/defaults/
"""

FeatureMetadataDict = Dict[str, Any]
"""Metadata for a specific feature like idx2str."""

FeaturePostProcessingOutputDict = Dict[str, Any]
"""Output from feature post-processing."""
