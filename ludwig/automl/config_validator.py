# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-flight validation that a Ludwig config is compatible with a DataFrame."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ludwig.api_annotations import DeveloperAPI
from ludwig.automl.config_enumerator import (
    FeatureSpec,
    get_valid_combiners,
    get_valid_decoders,
    get_valid_encoders,
)
from ludwig.constants import (
    BATCH_SIZE,
    COMBINER,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    TYPE,
)

logger = logging.getLogger(__name__)

# Fraction of the dataset assumed to go to the training split for the batch-size check.
_TRAIN_SPLIT_FRACTION: float = 0.7


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@DeveloperAPI
@dataclass
class ValidationResult:
    """Result of a pre-flight config validation.

    :attr is_valid: ``True`` when all checks passed (and, if ``strict=True``, no warnings either).
    :attr failures: Hard errors that definitely make the config incompatible.
    :attr warnings: Soft issues that *may* cause problems but are not necessarily fatal.
    """

    is_valid: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_feature_names(features: list[dict]) -> list[str]:
    """Extracts the ``name`` key from each feature dict."""
    return [f.get("name", "") for f in features]


def _get_feature_type(feature: dict) -> str | None:
    """Returns the ``type`` field of a feature dict, or ``None`` if absent."""
    return feature.get(TYPE)


def _get_encoder_type(feature: dict) -> str | None:
    """Returns the encoder type nested inside a feature dict, or ``None`` if absent."""
    encoder = feature.get("encoder")
    if isinstance(encoder, dict):
        return encoder.get(TYPE)
    if isinstance(encoder, str):
        return encoder
    return None


def _get_decoder_type(feature: dict) -> str | None:
    """Returns the decoder type nested inside a feature dict, or ``None`` if absent."""
    decoder = feature.get("decoder")
    if isinstance(decoder, dict):
        return decoder.get(TYPE)
    if isinstance(decoder, str):
        return decoder
    return None


def _get_combiner_type(config_dict: dict) -> str | None:
    """Returns the combiner type from a config dict, or ``None`` if absent."""
    combiner = config_dict.get(COMBINER)
    if isinstance(combiner, dict):
        return combiner.get(TYPE)
    if isinstance(combiner, str):
        return combiner
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@DeveloperAPI
def validate_config_for_dataset(
    config_dict: dict,
    df: pd.DataFrame,
    strict: bool = False,
) -> ValidationResult:
    """Pre-flight validation that a Ludwig config is compatible with a DataFrame.

    Performs the following checks **without training**:

    1. All ``input_features`` columns exist in *df*.
    2. All ``output_features`` columns exist in *df*.
    3. Encoder types are valid for their respective feature types.
    4. Decoder types are valid for their respective output feature types.
    5. Combiner type is valid for the input feature set.
    6. ``batch_size`` (if specified) is less than ``len(df) * 0.7`` so the training
       split contains at least one full batch.
    7. No output feature column is also an input feature column.
    8. Each output feature column has more than 1 distinct non-null value.

    # Inputs
    :param config_dict: (dict) A Ludwig config dict (as produced by
        :func:`~ludwig.automl.config_sampler.sample_configs` or hand-crafted).
    :param df: (pd.DataFrame) The dataset the config will be trained on.
    :param strict: (bool) When ``True``, WARN-level issues also cause
        :attr:`ValidationResult.is_valid` to be ``False``.

    # Return
    :return: (ValidationResult) Populated with any failures and/or warnings found.
    """
    failures: list[str] = []
    warnings: list[str] = []

    df_columns: set[str] = set(df.columns)
    input_features: list[dict] = config_dict.get(INPUT_FEATURES, [])
    output_features: list[dict] = config_dict.get(OUTPUT_FEATURES, [])

    input_names = _get_feature_names(input_features)
    output_names = _get_feature_names(output_features)

    # ------------------------------------------------------------------
    # Input/output columns present in dataframe
    # ------------------------------------------------------------------
    for name in input_names:
        if name not in df_columns:
            failures.append(f"Input feature column '{name}' not found in dataframe (columns: {sorted(df_columns)}).")

    for name in output_names:
        if name not in df_columns:
            failures.append(f"Output feature column '{name}' not found in dataframe (columns: {sorted(df_columns)}).")

    # ------------------------------------------------------------------
    # Encoder types valid for their feature types
    # ------------------------------------------------------------------
    for feat in input_features:
        feat_name = feat.get("name", "<unknown>")
        feat_type = _get_feature_type(feat)
        enc_type = _get_encoder_type(feat)

        if feat_type is None:
            warnings.append(f"Input feature '{feat_name}' has no 'type' specified; skipping encoder check.")
            continue

        if enc_type is None:
            # No encoder specified — Ludwig will use the default, which is always valid.
            continue

        valid_encoders = get_valid_encoders(feat_type)
        if not valid_encoders:
            warnings.append(
                f"Input feature '{feat_name}' has unknown feature type '{feat_type}'; cannot validate encoder."
            )
        elif enc_type not in valid_encoders:
            failures.append(
                f"Encoder '{enc_type}' is not valid for feature type '{feat_type}' "
                f"(feature '{feat_name}'). Valid encoders: {valid_encoders}."
            )

    # ------------------------------------------------------------------
    # Decoder types valid for their output feature types
    # ------------------------------------------------------------------
    for feat in output_features:
        feat_name = feat.get("name", "<unknown>")
        feat_type = _get_feature_type(feat)
        dec_type = _get_decoder_type(feat)

        if feat_type is None:
            warnings.append(f"Output feature '{feat_name}' has no 'type' specified; skipping decoder check.")
            continue

        if dec_type is None:
            continue

        valid_decoders = get_valid_decoders(feat_type)
        if not valid_decoders:
            warnings.append(
                f"Output feature '{feat_name}' has unknown feature type '{feat_type}'; cannot validate decoder."
            )
        elif dec_type not in valid_decoders:
            failures.append(
                f"Decoder '{dec_type}' is not valid for output feature type '{feat_type}' "
                f"(feature '{feat_name}'). Valid decoders: {valid_decoders}."
            )

    # ------------------------------------------------------------------
    # Combiner optional-dependency check
    # ------------------------------------------------------------------
    combiner_type = _get_combiner_type(config_dict)
    _COMBINER_OPTIONAL_DEPS: dict[str, str] = {
        "tabpfn_v2": "tabpfn",
    }
    if combiner_type in _COMBINER_OPTIONAL_DEPS:
        dep = _COMBINER_OPTIONAL_DEPS[combiner_type]
        try:
            __import__(dep)
        except ImportError:
            failures.append(
                f"Combiner '{combiner_type}' requires the '{dep}' package which is not installed (pip install {dep})."
            )

    # ------------------------------------------------------------------
    # Combiner compatible with input feature set
    # ------------------------------------------------------------------
    combiner_type = _get_combiner_type(config_dict)
    if combiner_type is not None:
        # Build FeatureSpec list for features whose types are known.
        known_input_specs: list[FeatureSpec] = []
        for feat in input_features:
            ft = _get_feature_type(feat)
            fn = feat.get("name", "")
            if ft:
                known_input_specs.append(FeatureSpec(name=fn, type=ft))

        if known_input_specs:
            valid_combiners = get_valid_combiners(known_input_specs)
            if combiner_type not in valid_combiners:
                failures.append(
                    f"Combiner '{combiner_type}' is not compatible with the input feature schema "
                    f"(input types: {[s.type for s in known_input_specs]}). "
                    f"Valid combiners: {valid_combiners}."
                )
    else:
        warnings.append("No combiner type specified in config; skipping combiner compatibility check.")

    # ------------------------------------------------------------------
    # Batch size fits within training split
    # ------------------------------------------------------------------
    trainer = config_dict.get("trainer", {})
    batch_size = trainer.get(BATCH_SIZE) if isinstance(trainer, dict) else None

    if batch_size is not None and isinstance(batch_size, int):
        train_rows = len(df) * _TRAIN_SPLIT_FRACTION
        if batch_size >= train_rows:
            warnings.append(
                f"batch_size={batch_size} >= estimated training rows ({train_rows:.0f}). "
                "The training split may contain fewer than one full batch."
            )

    # ------------------------------------------------------------------
    # No column serves as both input and output
    # ------------------------------------------------------------------
    overlap = set(input_names) & set(output_names)
    if overlap:
        failures.append(
            f"The following column(s) appear in both input_features and output_features: {sorted(overlap)}."
        )

    # ------------------------------------------------------------------
    # Each output feature has enough distinct values to train on
    # ------------------------------------------------------------------
    for name in output_names:
        if name not in df_columns:
            # Already reported in check 2; skip to avoid duplicate noise.
            continue
        series = df[name].dropna()
        n_distinct = series.nunique()
        if n_distinct <= 1:
            failures.append(
                f"Output feature column '{name}' has {n_distinct} distinct non-null value(s); "
                "at least 2 are required for meaningful training."
            )

    # ------------------------------------------------------------------
    # Determine overall validity
    # ------------------------------------------------------------------
    is_valid = len(failures) == 0 and (not strict or len(warnings) == 0)

    result = ValidationResult(is_valid=is_valid, failures=failures, warnings=warnings)

    if failures:
        logger.warning(f"validate_config_for_dataset: {len(failures)} failure(s) found.")
        for msg in failures:
            logger.warning(f"  FAIL: {msg}")
    if warnings:
        logger.debug(f"validate_config_for_dataset: {len(warnings)} warning(s) found.")
        for msg in warnings:
            logger.debug(f"  WARN: {msg}")

    return result
