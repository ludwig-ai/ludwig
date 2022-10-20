# Copyright (c) 2022 Predibase, Inc.
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

from typing import Any

import numpy as np


def _dict_like(x):
    """Returns true if an object is a dict or convertible to one, false if not."""
    try:
        _ = dict(x)
    except (TypeError, ValueError):
        return False
    return True


def _enumerable(x):
    """Returns true if an object is enumerable, false if not."""
    try:
        _ = enumerate(x)
    except (TypeError, ValueError):
        return False
    return True


def assert_all_finite(x: Any, keypath=""):
    """Ensures that all scalars at all levels of the dictionary, list, array, or scalar are finite.

    keypath is only used for logging error messages, to indicate where the non-finite value was detected.
    """
    path_description = f" at {keypath} " if keypath else " "
    if np.isscalar(x):
        assert np.isfinite(x), f"Value{path_description}should be finite, but is {str(x)}."
    elif isinstance(x, np.ndarray):
        non_finite_indices = np.nonzero(~np.isfinite(x))
        non_finite_values = x[non_finite_indices]
        assert np.all(np.isfinite(x)), (
            f"All values{path_description}should be finite, but found {str(non_finite_values)} "
            "at positions {str(np.array(non_finite_indices).flatten())}."
        )
    elif _dict_like(x):
        # x is either a dict or convertible to one
        for k, v in dict(x).items():
            assert_all_finite(v, keypath=keypath + "." + str(k) if keypath else str(k))
    elif _enumerable(x):
        # x is a list, set or other enumerable type, but not a string, dict, or numpy array.
        for i, v in enumerate(x):
            assert_all_finite(v, keypath=keypath + f"[{i}]")
    else:
        assert False, f"Unhandled type {str(type(x))} for value{path_description}"
