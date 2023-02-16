#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ludwig.constants import NAME, PREPROCESSING, SEQUENCE, TEXT, TIMESERIES, TYPE
from ludwig.utils.data_utils import hash_dict
from ludwig.utils.strings_utils import get_tokenizer_from_registry, UNKNOWN_SYMBOL

SEQUENCE_TYPES = {SEQUENCE, TEXT, TIMESERIES}
FEATURE_NAME_SUFFIX = "__ludwig"
FEATURE_NAME_SUFFIX_LENGTH = len(FEATURE_NAME_SUFFIX)


def should_regularize(regularize_layers):
    regularize = False
    if isinstance(regularize_layers, bool) and regularize_layers:
        regularize = True
    elif isinstance(regularize_layers, (list, tuple)) and regularize_layers and regularize_layers[-1]:
        regularize = True
    return regularize


def set_str_to_idx(set_string, feature_dict, tokenizer_name):
    try:
        tokenizer = get_tokenizer_from_registry(tokenizer_name)()
    except ValueError:
        raise Exception(f"Tokenizer {tokenizer_name} not supported")

    out = [feature_dict.get(item, feature_dict[UNKNOWN_SYMBOL]) for item in tokenizer(set_string)]

    return np.array(out, dtype=np.int32)


def compute_token_probabilities(
    probabilities: Union[list, tuple, np.ndarray],
) -> np.ndarray:
    """Gets the maximum probability per timestep.

    Args:
        probabilities: An iterable of iterables or np.ndarray with shape (sequence_length, num_classes)
            where each inner iterable or np.ndarray is the probability distribution for a single timestep.
    Returns:
        An np.ndarray with shape (sequence_length,) containing the maximum probability for each timestep.
    """
    if isinstance(probabilities, (list, tuple)):
        max_probs = []
        for timestep_probs in probabilities:
            max_probs.append(np.max(timestep_probs))
        max_probs = np.array(max_probs)
    elif isinstance(probabilities, np.ndarray):
        max_probs = np.max(probabilities, axis=-1)
    else:
        raise ValueError(f"probabilities type must be in [list, tuple, np.ndarray]. Got {type(probabilities)}")
    return max_probs


def compute_sequence_probability(
    sequence_probabilities: np.ndarray,
    max_sequence_length: Optional[int] = None,
    return_log_prob: bool = True,
) -> float:
    """Computes the sequence level probability.

    Args:
        sequence_probabilities: An iterable of iterables or np.ndarray with shape (sequence_length,)
        max_sequence_length: The maximum sequence length to use. If None, uses the first dim of `sequence_probabilities`
        return_log_prob: Whether to return the log probability. Defaults to True.
    """
    if max_sequence_length is None:
        max_sequence_length = sequence_probabilities.shape[0]

    sequence_probabilities = sequence_probabilities[:max_sequence_length]

    if return_log_prob:
        return np.sum(np.log(sequence_probabilities))
    else:
        return np.prod(sequence_probabilities)


def sanitize(name):
    """Replaces invalid id characters."""
    return re.sub("\\W|^(?=\\d)", "_", name)


def compute_feature_hash(feature: dict) -> str:
    """This function computes a hash for each feature based on the preprocessing dictionary associated with each
    feature, as well as the feature's type.

    Args:
        feature: Feature dictionary

    Returns: Feature hash name
    """
    feature_data = dict(
        preprocessing=feature.get(PREPROCESSING, {}),
        type=feature[TYPE],
    )
    return sanitize(feature[NAME]) + "_" + hash_dict(feature_data).decode("ascii")


def get_input_size_with_dependencies(
    combiner_output_size: int, dependencies: List[str], other_output_features  # Dict[str, "OutputFeature"]
):
    """Returns the input size for the first layer of this output feature's FC stack, accounting for dependencies on
    other output features.

    In the forward pass, the hidden states of any dependent output features get concatenated with the combiner's output.
    If this output feature depends on other output features, then the input size for this feature's FCStack is the sum
    of the output sizes of other output features + the combiner's output size.
    """
    input_size_with_dependencies = combiner_output_size
    for feature_name in dependencies:
        if other_output_features[feature_name].fc_stack.num_layers:
            input_size_with_dependencies += other_output_features[feature_name].fc_stack.output_shape[-1]
        else:
            # 0-layer FCStack. Use the output feature's input size.
            input_size_with_dependencies += other_output_features[feature_name].input_size
    return input_size_with_dependencies


def get_module_dict_key_from_name(name: str, feature_name_suffix: str = FEATURE_NAME_SUFFIX) -> str:
    """Returns a key that's guaranteed to be compatible with torch."""
    key = name.replace(".", "__ludwig_punct_period__")
    return key + feature_name_suffix


def get_name_from_module_dict_key(key: str, feature_name_suffix_length: int = FEATURE_NAME_SUFFIX_LENGTH) -> str:
    """Reverse of get_module_dict_key_from_name."""
    name = key.replace("__ludwig_punct_period__", ".")
    return name[:-feature_name_suffix_length]


class LudwigFeatureDict(torch.nn.Module):
    """Torch ModuleDict wrapper that permits keys with any name.

    Torch's ModuleDict implementation doesn't allow certain keys to be used if they conflict with existing class
    attributes, e.g.

    > torch.nn.ModuleDict({'type': torch.nn.Module()})  # Raises KeyError.

    This class is a simple wrapper around torch's ModuleDict that mitigates possible conflicts by using a key-suffixing
    protocol.

    This is also tracked in Pytorch: https://github.com/pytorch/pytorch/issues/71203.
    """

    def __init__(self):
        super().__init__()
        self.module_dict = torch.nn.ModuleDict()
        self.internal_key_to_original_name_map = {}

    def __getitem__(self, key) -> torch.nn.Module:
        return self.module_dict[get_module_dict_key_from_name(key)]

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        module_dict_key_name = get_module_dict_key_from_name(key)
        self.internal_key_to_original_name_map[module_dict_key_name] = key
        self.module_dict[module_dict_key_name] = module

    def __len__(self) -> int:
        return len(self.module_dict)

    def __next__(self) -> None:
        return next(iter(self))

    def __iter__(self) -> None:
        return iter(self.keys())

    def keys(self) -> List[str]:
        return [
            get_name_from_module_dict_key(feature_name)
            for feature_name in self.internal_key_to_original_name_map.keys()
        ]

    def values(self) -> List[torch.nn.Module]:
        return [module for _, module in self.module_dict.items()]

    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return [
            (get_name_from_module_dict_key(feature_name), module) for feature_name, module in self.module_dict.items()
        ]

    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        for feature_name, module in modules.items():
            self.__setitem__(feature_name, module)
