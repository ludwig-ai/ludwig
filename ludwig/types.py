"""Public API: Common typing for Ludwig dictionary parameters."""

from typing import Any, Dict

FeatureConfigDict = dict[str, Any]
"""Dictionary of parameters used to configure an input or output feature.

https://ludwig.ai/latest/configuration/features/supported_data_types/
"""

ModelConfigDict = dict[str, Any]
"""Dictionary representation of the ModelConfig object.

https://ludwig.ai/latest/configuration/
"""

TrainingSetMetadataDict = dict[str, Any]
"""Training set metadata, which consists of internal configuration parameters."""

PreprocessingConfigDict = dict[str, Any]
"""Dictionary of parameters used to configure preprocessing.

May be type-defaults global preprocessing or feature-specific preprocessing.
https://ludwig.ai/latest/configuration/preprocessing/
"""

HyperoptConfigDict = dict[str, Any]
"""Dictionary of parameters used to configure hyperopt.

https://ludwig.ai/latest/configuration/hyperparameter_optimization/
"""

TrainerConfigDict = dict[str, Any]
"""Dictionary of parameters used to configure training.

https://ludwig.ai/latest/configuration/trainer/
"""

FeatureTypeDefaultsDict = dict[str, FeatureConfigDict]
"""Dictionary of type to parameters that configure the defaults for that feature type.

https://ludwig.ai/latest/configuration/defaults/
"""

FeatureMetadataDict = dict[str, Any]
"""Metadata for a specific feature like idx2str."""

FeaturePostProcessingOutputDict = dict[str, Any]
"""Output from feature post-processing."""
