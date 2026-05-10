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
"""Enumerates all valid (encoder, combiner, decoder) combinations for a given feature schema."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import NamedTuple

from ludwig.api_annotations import DeveloperAPI
from ludwig.automl.search_space import _default_search_space, SearchSpace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backwards-compatible module-level registry aliases
# ---------------------------------------------------------------------------
# These are computed lazily from the default SearchSpace so that existing code
# that does ``from ludwig.automl.config_enumerator import ENCODER_REGISTRY``
# continues to work without modification.


def __getattr__(name: str):
    if name == "ENCODER_REGISTRY":
        return _default_search_space().encoder_registry
    if name == "DECODER_REGISTRY":
        return _default_search_space().decoder_registry
    if name == "ALL_COMBINERS":
        return list(_default_search_space().combiners.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@DeveloperAPI
class FeatureSpec(NamedTuple):
    """Minimal description of a feature: its column name and Ludwig feature type."""

    name: str
    type: str  # Ludwig feature type constant


@DeveloperAPI
@dataclass
class ConfigSpec:
    """A single valid (encoder assignments, combiner, decoder) combination."""

    input_encoders: dict[str, str]  # feature_name -> encoder_type
    combiner: str
    output_decoder: str
    output_type: str


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


@DeveloperAPI
def get_valid_encoders(feature_type: str, search_space: SearchSpace | None = None) -> list[str]:
    """Returns the list of valid encoder names for *feature_type*.

    # Inputs
    :param feature_type: (str) A Ludwig feature type constant (e.g. ``"text"``, ``"image"``).
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[str]) Encoder names registered for that feature type.  Returns an empty
        list for unknown feature types (with a warning logged).
    """
    ss = search_space or _default_search_space()
    encoders = ss.encoder_registry.get(feature_type)
    if encoders is None:
        logger.warning(f"No encoder registry entry for feature type '{feature_type}'; returning empty list.")
        return []
    return list(encoders)


@DeveloperAPI
def get_valid_decoders(feature_type: str, search_space: SearchSpace | None = None) -> list[str]:
    """Returns the list of valid decoder names for *feature_type*.

    # Inputs
    :param feature_type: (str) A Ludwig feature type constant used as an output feature.
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[str]) Decoder names registered for that feature type.  Returns an empty
        list for unknown/unsupported output feature types (with a warning logged).
    """
    ss = search_space or _default_search_space()
    decoders = ss.decoder_registry.get(feature_type)
    if decoders is None:
        logger.warning(f"No decoder registry entry for feature type '{feature_type}'; returning empty list.")
        return []
    return list(decoders)


@DeveloperAPI
def get_valid_combiners(
    input_features: list[FeatureSpec],
    search_space: SearchSpace | None = None,
) -> list[str]:
    """Returns the list of valid combiner names for the given input feature schema.

    Applies compatibility constraints stored in each :class:`~ludwig.automl.search_space.CombinerSpec`:

    - ``requires_all_tabular``: only when **all** input features are tabular
      (BINARY, CATEGORY, or NUMBER).
    - ``exact_n_inputs``: only when the number of input features equals that value.
    - ``requires_sequential``: only when at least one input feature is
      SEQUENCE, TEXT, or TIMESERIES.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[str]) Compatible combiner names.
    """
    ss = search_space or _default_search_space()

    _tabular = frozenset({"binary", "category", "number"})
    _sequential = frozenset({"sequence", "text", "timeseries"})

    input_types = {f.type for f in input_features}
    n_inputs = len(input_features)
    all_tabular = input_types.issubset(_tabular)
    has_sequential = bool(input_types & _sequential)

    valid: list[str] = []
    for spec in ss.combiners.values():
        c = spec.constraints
        if c.get("requires_all_tabular") and not all_tabular:
            continue
        exact_n = c.get("exact_n_inputs")
        if exact_n is not None and n_inputs != exact_n:
            continue
        if c.get("requires_sequential") and not has_sequential:
            continue
        valid.append(spec.name)

    return valid


# ---------------------------------------------------------------------------
# Full enumeration
# ---------------------------------------------------------------------------


@DeveloperAPI
def enumerate_config_specs(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    max_configs: int | None = None,
    search_space: SearchSpace | None = None,
) -> list[ConfigSpec]:
    """Enumerates all valid (encoder, combiner, decoder) combinations for the given schema.

    Each :class:`ConfigSpec` specifies one encoder per input feature, one combiner, and one
    decoder for the output feature.  The combinatorial space can be very large; *max_configs*
    caps the output using deterministic sampling across all axes.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.
    :param output_feature: (FeatureSpec) The single output feature specification.
    :param max_configs: (int | None) Maximum number of ``ConfigSpec`` objects to return.
        ``None`` means unlimited.
    :param search_space: (:class:`~ludwig.automl.search_space.SearchSpace` | None) Optional
        custom search space.  Uses the built-in defaults when ``None``.

    # Return
    :return: (list[ConfigSpec]) All (or up to *max_configs*) valid config specs.
    """
    ss = search_space or _default_search_space()

    per_feature_encoders: list[list[str]] = [get_valid_encoders(f.type, ss) for f in input_features]
    valid_combiners = get_valid_combiners(input_features, ss)
    valid_decoders = get_valid_decoders(output_feature.type, ss)

    if not valid_combiners:
        logger.warning("No valid combiners for the given input feature schema; returning empty list.")
        return []
    if not valid_decoders:
        logger.warning(f"No valid decoders for output feature type '{output_feature.type}'; returning empty list.")
        return []

    total = 1
    for enc_list in per_feature_encoders:
        total *= max(len(enc_list), 1)
    total *= len(valid_combiners) * len(valid_decoders)

    logger.debug(
        f"enumerate_config_specs: {total} total combinations "
        f"(features={len(input_features)}, combiners={len(valid_combiners)}, decoders={len(valid_decoders)})"
    )

    results: list[ConfigSpec] = []
    encoder_combos = itertools.product(*per_feature_encoders)

    if max_configs is not None and total > max_configs:
        stride = max(1, total // max_configs)
        flat_idx = 0
        next_sample = 0
        for enc_combo in encoder_combos:
            for combiner in valid_combiners:
                for decoder in valid_decoders:
                    if flat_idx == next_sample:
                        results.append(
                            ConfigSpec(
                                input_encoders={f.name: enc for f, enc in zip(input_features, enc_combo)},
                                combiner=combiner,
                                output_decoder=decoder,
                                output_type=output_feature.type,
                            )
                        )
                        next_sample += stride
                        if len(results) >= max_configs:
                            return results
                    flat_idx += 1
    else:
        for enc_combo in encoder_combos:
            for combiner in valid_combiners:
                for decoder in valid_decoders:
                    results.append(
                        ConfigSpec(
                            input_encoders={f.name: enc for f, enc in zip(input_features, enc_combo)},
                            combiner=combiner,
                            output_decoder=decoder,
                            output_type=output_feature.type,
                        )
                    )

    return results
