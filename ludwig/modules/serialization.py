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
import json
import logging
from functools import lru_cache
from typing import Any, Dict

import h5py
import numpy as np

from ludwig.globals import LUDWIG_VERSION
from ludwig.modules.ludwig_module import LudwigModule, LudwigModuleState, module_registry
from ludwig.utils.fs_utils import download_h5, upload_h5

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Python json library does not support serialization of numpy types.

    This custom encoder converts numpy scalars to python types.
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


def _save_state_to_group(state: LudwigModuleState, group: h5py.Group):
    """Save LudwigModuleState to HDF5 group, recursively creating groups for any child modules."""
    group.attrs["type"] = state.type
    group.attrs["ludwig_version"] = state.ludwig_version
    group.attrs["config"] = json.dumps(state.config, cls=NumpyEncoder, sort_keys=True)
    group.attrs["metadata"] = json.dumps(state.metadata, cls=NumpyEncoder, sort_keys=True)
    for k, w in state.saved_weights.items():
        group.create_dataset(k, data=w)
    for k, child in state.children.items():
        child_group = group.create_group(k)
        _save_state_to_group(child, child_group)


def save_state_to_file(state: LudwigModuleState, url_or_path: str):
    """Serializes LudwigModuleState to file."""
    with upload_h5(url_or_path) as f:
        # with h5py.File(f, "w") as f:
        _save_state_to_group(state, f)


def _load_state_from_group(group) -> LudwigModuleState:
    """Restore LudwigModuleState from HDF5 group."""
    return LudwigModuleState(
        type=group.attrs["type"],
        ludwig_version=group.attrs["ludwig_version"],
        config=json.loads(group.attrs["config"]),
        metadata=json.loads(group.attrs["metadata"]),
        saved_weights={k: v[()] for k, v in group.items() if isinstance(v, h5py.Dataset)},
        children={k: _load_state_from_group(v) for k, v in group.items() if isinstance(v, h5py.Group)},
    )


@lru_cache(maxsize=3)
def load_state_from_file(path_or_uri: str) -> LudwigModuleState:
    """Loads Ludwig Module state from a file."""
    with download_h5(path_or_uri) as f:
        # with h5py.File(f, "r") as f:
        # The file object does double duty as the HDF5 root group
        state = _load_state_from_group(f)
    return update_state_for_current_version(state)


def instantiate_module_from_state(state: LudwigModuleState, device: str = None) -> LudwigModule:
    """Instatiates a module by restoring from saved state."""
    cls = module_registry.get(state.type)
    if cls is None:
        logger.error(f"No Ludwig Module registered for name {state.type}")
        return None
    if not (hasattr(cls, "restore_from_state") and callable(cls.restore_from_state)):
        logger.error(f"The @classmethod restore_from_state must be implemented to restore {state.type} objects")
        return None
    restored_module = cls.restore_from_state(state)
    if restored_module is not None and device is not None:
        restored_module = restored_module.to(device)
    return restored_module


def save(object: LudwigModule, path_or_uri: str, metadata: Dict[str, Any] = None):
    """Saves Ludwig object to file or buffer."""
    object_state = object.get_state(metadata=metadata)
    save_state_to_file(object_state, path_or_uri)


def load(path_or_uri: str, device: str = None) -> LudwigModule:
    """Loads saved Ludwig module from file or buffer. If the module has parameters which are torch Tensors, device
    specifies the device where tensors will be instantiated.

    Args:
        f: The file path or object to load from.
        device: 'cuda' or 'cpu'
    """
    state = load_state_from_file(path_or_uri)
    return instantiate_module_from_state(state, device=device)


def update_state_for_current_version(state: LudwigModuleState) -> LudwigModuleState:
    """Update saved Ludwig object from previous version."""
    if state == LUDWIG_VERSION:
        return state
    else:
        # TODO: check version, apply updates if needed.
        state.ludwig_version = LUDWIG_VERSION
        return state
