"""Public API: Common typing for Ludwig dictionary parameters.

These TypedDicts document the shape of the dicts flowing through Ludwig's
public surface.  They use ``total=False`` so that callers can omit optional
keys without triggering type errors.  The legacy ``dict[str, Any]`` aliases
are kept for backward compatibility but are deprecated — prefer the TypedDicts.
"""

from __future__ import annotations

from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------


class FeatureConfigDict(TypedDict, total=False):
    """Parameters used to configure a single input or output feature.

    See https://ludwig.ai/latest/configuration/features/supported_data_types/
    """

    name: str
    type: str
    column: str
    tied: str | None
    encoder: dict[str, Any]
    decoder: dict[str, Any]
    preprocessing: dict[str, Any]
    loss: dict[str, Any]
    output_size: int
    num_fc_layers: int
    fc_layers: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class ModelConfigDict(TypedDict, total=False):
    """Dictionary representation of the ModelConfig object.

    See https://ludwig.ai/latest/configuration/
    """

    model_type: str
    input_features: list[FeatureConfigDict]
    output_features: list[FeatureConfigDict]
    combiner: dict[str, Any]
    trainer: dict[str, Any]
    preprocessing: dict[str, Any]
    defaults: dict[str, Any]
    hyperopt: dict[str, Any]
    backend: dict[str, Any]
    ludwig_version: str
    preset: str


# ---------------------------------------------------------------------------
# Training set metadata
# ---------------------------------------------------------------------------


class FeatureMetadataDict(TypedDict, total=False):
    """Metadata for a single feature, produced during preprocessing.

    Contents are feature-type-specific; common keys are listed here.
    """

    idx2str: list[str]
    str2idx: dict[str, int]
    str2freq: dict[str, int]
    vocab_size: int
    max_sequence_length: int
    reshape: list[int] | None
    mean: float
    std: float
    min: float
    max: float
    missing_value_strategy: str
    computed_fill_value: float | str | None
    lazy: bool
    mode: str
    prefetch_size: int | None
    lazy_audio_params: dict[str, Any]
    lazy_image_params: dict[str, Any]


class TrainingSetMetadataDict(TypedDict, total=False):
    """Training set metadata produced during preprocessing and saved alongside the dataset cache.

    Top-level keys are feature names; values are :class:`FeatureMetadataDict`.
    Global keys (e.g. ``preprocessing_parameters``) are also present.
    """

    preprocessing_parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Preprocessing / trainer / hyperopt config dicts
# ---------------------------------------------------------------------------


class PreprocessingConfigDict(TypedDict, total=False):
    """Parameters used to configure preprocessing (global or per-feature).

    See https://ludwig.ai/latest/configuration/preprocessing/
    """

    split: dict[str, Any]
    sample_ratio: float
    oversample_minority: float | None
    undersample_majority: float | None


class TrainerConfigDict(TypedDict, total=False):
    """Parameters used to configure training.

    See https://ludwig.ai/latest/configuration/trainer/
    """

    type: str
    epochs: int
    batch_size: int | str
    learning_rate: float | str
    optimizer: dict[str, Any]
    regularization_type: str | None
    regularization_lambda: float
    gradient_clipping: dict[str, Any]
    eval_steps: int
    early_stop: int
    steps_per_checkpoint: int


class HyperoptConfigDict(TypedDict, total=False):
    """Parameters used to configure hyperparameter optimisation.

    See https://ludwig.ai/latest/configuration/hyperparameter_optimization/
    """

    executor: dict[str, Any]
    search_alg: dict[str, Any]
    parameters: dict[str, Any]
    goal: str
    metric: str
    output_feature: str
    split: str


# ---------------------------------------------------------------------------
# Misc composite dicts
# ---------------------------------------------------------------------------


FeatureTypeDefaultsDict = dict[str, FeatureConfigDict]
"""Dictionary mapping feature type name → default FeatureConfigDict.

See https://ludwig.ai/latest/configuration/defaults/
"""

FeaturePostProcessingOutputDict = dict[str, Any]
"""Output from feature post-processing (feature-type-specific shapes)."""
