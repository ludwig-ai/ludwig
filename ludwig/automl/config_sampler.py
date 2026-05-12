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
"""Samples diverse Ludwig configs for a given feature schema."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass

import pandas as pd

from ludwig.api_annotations import DeveloperAPI
from ludwig.automl.config_enumerator import (
    ConfigSpec,
    FeatureSpec,
    get_valid_combiners,
    get_valid_decoders,
    get_valid_encoders,
)
from ludwig.automl.search_space import _default_search_space, SearchSpace
from ludwig.constants import (
    BATCH_SIZE,
    COMBINER,
    INPUT_FEATURES,
    LEARNING_RATE,
    OUTPUT_FEATURES,
    TRAINER,
    TYPE,
)

logger = logging.getLogger(__name__)

# Maximum number of encoder choices sampled per feature type to keep configs tractable.
_MAX_ENCODERS_PER_TYPE: int = 3


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@DeveloperAPI
@dataclass
class SampledConfig:
    """A complete Ludwig-style config dict ready for training."""

    config_dict: dict  # Full Ludwig config as dict
    config_hash: str  # SHA-256 of canonical JSON
    spec: ConfigSpec
    trainer_params: dict  # LR, batch_size, etc.
    seed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: dict) -> str:
    """Returns a deterministic JSON string suitable for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _hash_config(config_dict: dict) -> str:
    """Returns the SHA-256 hex digest of the canonical JSON representation of *config_dict*."""
    return hashlib.sha256(_canonical_json(config_dict).encode()).hexdigest()


def _build_config_dict(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    spec: ConfigSpec,
    trainer_params: dict,
    combiner_params: dict | None = None,
    encoder_hyperparams: dict[str, dict] | None = None,
) -> dict:
    """Builds a complete Ludwig config dict from a :class:`ConfigSpec` and trainer params.

    # Inputs
    :param input_features: Input feature specs.
    :param output_feature: Output feature spec.
    :param spec: The :class:`ConfigSpec` (encoder assignments, combiner, decoder).
    :param trainer_params: Trainer hyperparameters dict.
    :param combiner_params: Optional combiner hyperparameter dict (includes ``type`` key).
    :param encoder_hyperparams: Optional mapping of feature name -> encoder hyperparam dict
        (merged into each input feature's ``encoder`` sub-dict).
    """
    encoder_hyperparams = encoder_hyperparams or {}
    input_feature_dicts = []
    for f in input_features:
        enc_dict: dict = {TYPE: spec.input_encoders[f.name]}
        extra = encoder_hyperparams.get(f.name, {})
        # extra already contains {type: ..., param: val, ...} — merge, type wins from spec
        for k, v in extra.items():
            if k != TYPE:
                enc_dict[k] = v
        input_feature_dicts.append({"name": f.name, TYPE: f.type, "encoder": enc_dict})

    output_feature_dict = {
        "name": output_feature.name,
        TYPE: output_feature.type,
        "decoder": {TYPE: spec.output_decoder},
    }
    combiner_dict = combiner_params if combiner_params is not None else {TYPE: spec.combiner}
    return {
        INPUT_FEATURES: input_feature_dicts,
        OUTPUT_FEATURES: [output_feature_dict],
        COMBINER: combiner_dict,
        TRAINER: dict(trainer_params),
    }


def _sample_trainer_params(
    rng: random.Random,
    max_epochs: int,
    time_limit_s: int | None,
    search_space: SearchSpace | None = None,
) -> dict:
    """Samples a single set of trainer hyperparameters from the search space grids."""
    ss = search_space or _default_search_space()
    params: dict = {
        LEARNING_RATE: rng.choice(ss.trainer.learning_rate_values),
        BATCH_SIZE: rng.choice(ss.trainer.batch_size_values),
        "epochs": max_epochs,
    }
    if time_limit_s is not None:
        params["time_limit_s"] = time_limit_s
    return params


def _candidate_encoders(
    feature_type: str,
    rng: random.Random,
    search_space: SearchSpace | None = None,
) -> list[str]:
    """Returns up to *_MAX_ENCODERS_PER_TYPE* encoders for *feature_type*, sampled without replacement."""
    all_encoders = get_valid_encoders(feature_type, search_space)
    if len(all_encoders) <= _MAX_ENCODERS_PER_TYPE:
        return list(all_encoders)
    return rng.sample(all_encoders, _MAX_ENCODERS_PER_TYPE)


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------


@DeveloperAPI
def sample_configs(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    n: int = 100,
    seed: int = 42,
    max_epochs: int = 50,
    time_limit_s: int | None = None,
    search_space: SearchSpace | None = None,
) -> list[SampledConfig]:
    """Samples *n* diverse Ludwig configs for the given feature schema.

    Uses stratified sampling across combiners to ensure diversity:

    - At least 1 config per valid combiner (up to *n*).
    - Remaining configs distributed proportionally across combiners.
    - Trainer hyperparams sampled from the search space grids (LR, batch size).
    - Encoder and combiner hyperparams sampled from per-spec grids in the search space.
    - Encoder choices sampled per feature type (up to 3 options per type).
    - Configs are deduplicated by :attr:`SampledConfig.config_hash`.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.
    :param output_feature: (FeatureSpec) The single output feature specification.
    :param n: (int) Target number of sampled configs (after deduplication may be lower).
    :param seed: (int) Random seed for reproducibility.
    :param max_epochs: (int) Maximum training epochs written into each config's trainer section.
    :param time_limit_s: (int | None) Optional wall-clock time limit passed to the trainer.
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[SampledConfig]) Deduplicated sampled configs (up to *n*).
    """
    ss = search_space or _default_search_space()
    rng = random.Random(seed)

    valid_combiners = get_valid_combiners(input_features, ss)
    valid_decoders = get_valid_decoders(output_feature.type, ss)

    if not valid_combiners:
        logger.warning("sample_configs: no valid combiners for the given input feature schema.")
        return []
    if not valid_decoders:
        logger.warning(f"sample_configs: no valid decoders for output type '{output_feature.type}'.")
        return []

    # Build combiner budget: at least 1 per combiner, remainder distributed evenly.
    n_combiners = len(valid_combiners)
    base_per_combiner = max(1, n // n_combiners)
    remainder = max(0, n - base_per_combiner * n_combiners)

    combiner_budget: dict[str, int] = {}
    for i, c in enumerate(valid_combiners):
        combiner_budget[c] = base_per_combiner + (1 if i < remainder else 0)

    seen_hashes: set[str] = set()
    results: list[SampledConfig] = []

    for combiner, budget in combiner_budget.items():
        combiner_spec = ss.combiners.get(combiner)
        attempts = 0
        generated = 0
        max_attempts = budget * 20  # avoid infinite loop on very small spaces

        while generated < budget and attempts < max_attempts:
            attempts += 1

            # Sample one encoder per input feature (with hyperparams).
            input_encoders: dict[str, str] = {}
            encoder_hyperparams: dict[str, dict] = {}
            for feat in input_features:
                choices = _candidate_encoders(feat.type, rng, ss)
                if not choices:
                    logger.warning(f"sample_configs: no encoders for feature type '{feat.type}'; skipping feature.")
                    choices = ["passthrough"]
                enc_name = rng.choice(choices)
                input_encoders[feat.name] = enc_name
                enc_spec = ss.encoders.get(enc_name)
                if enc_spec is not None:
                    encoder_hyperparams[feat.name] = ss.sample_hyperparams(enc_spec, rng)

            decoder = rng.choice(valid_decoders)
            trainer_params = _sample_trainer_params(rng, max_epochs, time_limit_s, ss)

            # Sample combiner hyperparams via the SearchSpace.
            if combiner_spec is not None:
                combiner_params = ss.sample_hyperparams(combiner_spec, rng)
            else:
                combiner_params = {TYPE: combiner}

            spec = ConfigSpec(
                input_encoders=input_encoders,
                combiner=combiner,
                output_decoder=decoder,
                output_type=output_feature.type,
            )
            config_dict = _build_config_dict(
                input_features, output_feature, spec, trainer_params, combiner_params, encoder_hyperparams
            )
            config_hash = _hash_config(config_dict)

            if config_hash in seen_hashes:
                continue

            seen_hashes.add(config_hash)
            results.append(
                SampledConfig(
                    config_dict=config_dict,
                    config_hash=config_hash,
                    spec=spec,
                    trainer_params=trainer_params,
                    seed=seed,
                )
            )
            generated += 1

        if generated < budget:
            logger.debug(
                f"sample_configs: combiner '{combiner}' only produced {generated}/{budget} unique configs "
                f"after {attempts} attempts (search space may be small)."
            )

    logger.info(f"sample_configs: produced {len(results)} unique configs (target={n}).")
    return results[:n]


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


@DeveloperAPI
def configs_from_dataframe(
    df: pd.DataFrame,
    target_column: str,
    n: int = 100,
    seed: int = 42,
    search_space: SearchSpace | None = None,
) -> list[SampledConfig]:
    """Convenience function: infers the feature schema from *df* and samples *n* configs.

    Uses Ludwig's existing type-inference logic (:func:`~ludwig.utils.automl.type_inference.infer_type`)
    to determine feature types for every column.  The *target_column* is treated as the
    output feature; all other non-excluded columns become input features.

    # Inputs
    :param df: (pd.DataFrame) The dataset to infer types from.
    :param target_column: (str) Name of the output / target column.
    :param n: (int) Target number of sampled configs.
    :param seed: (int) Random seed for reproducibility.
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[SampledConfig]) Sampled configs for the inferred schema.
    """
    from ludwig.automl.base_config import convert_targets, get_dataset_info, get_field_metadata

    dataset_info = get_dataset_info(df)
    targets = convert_targets(target_column)
    metadata = get_field_metadata(dataset_info.fields, dataset_info.row_count, targets)

    input_features: list[FeatureSpec] = []
    output_feature: FeatureSpec | None = None

    for field_meta in metadata:
        if field_meta.name == target_column:
            output_feature = FeatureSpec(name=field_meta.name, type=field_meta.config.type)
        elif not field_meta.excluded and field_meta.mode == "input":
            input_features.append(FeatureSpec(name=field_meta.name, type=field_meta.config.type))

    if output_feature is None:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    if not input_features:
        raise ValueError("No valid input features could be inferred from the dataframe.")

    logger.info(
        f"configs_from_dataframe: inferred {len(input_features)} input features, "
        f"output='{output_feature.name}' (type={output_feature.type})."
    )

    # Scale down epochs and cap batch sizes for large datasets to avoid OOM and timeouts.
    # These thresholds are conservative; they keep wall time and VRAM use manageable on
    # a single consumer GPU (10–24 GiB) without sacrificing benchmark coverage.
    n_rows = len(df)
    max_epochs = 50
    if search_space is None and n_rows > 100_000:
        from ludwig.automl.search_space import _DEFAULT_SEARCH_SPACE_DIR, load_search_space, SearchSpace, TrainerSpec

        base = load_search_space(_DEFAULT_SEARCH_SPACE_DIR)
        if n_rows > 500_000:
            max_epochs = 5
            batch_sizes = [256, 512, 1024]
        else:
            max_epochs = 10
            batch_sizes = [128, 256, 512]
        search_space = SearchSpace(
            encoders=base.encoders,
            combiners=base.combiners,
            decoders=base.decoders,
            trainer=TrainerSpec(
                learning_rate_values=base.trainer.learning_rate_values,
                batch_size_values=batch_sizes,
            ),
        )

    return sample_configs(
        input_features=input_features,
        output_feature=output_feature,
        n=n,
        seed=seed,
        max_epochs=max_epochs,
        search_space=search_space,
    )
