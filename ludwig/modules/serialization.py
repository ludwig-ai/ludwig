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
import os
from typing import BinaryIO, IO, Union

import h5py
import numpy as np
import torch

from ludwig.globals import LUDWIG_VERSION
from ludwig.modules.ludwig_module import LudwigModule, LudwigModuleState


class NumpyEncoder(json.JSONEncoder):
    """Python json library does not support serialization of numpy types.

    This custom encoder converts numpy scalars to python types
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
    group.attrs["config"] = json.dumps(state.config, cls=NumpyEncoder)
    for k, w in state.saved_weights.items():
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        group.create_dataset(k, data=w)
    for k, child in state.children.items():
        child_group = group.create_group(k)
        _save_state_to_group(child, child_group)


def save_state_to_file(state: LudwigModuleState, f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    """Serializes LudwigModuleState to file."""
    with h5py.File(f, "w") as f:
        _save_state_to_group(state, f)


def _load_state_from_group(group) -> LudwigModuleState:
    """Restore LudwigModuleState from HDF5 group."""
    return LudwigModuleState(
        type=group.attrs["type"],
        ludwig_version=group.attrs["ludwig_version"],
        config=json.loads(group.attrs["config"]),
        saved_weights={},
        children={k: _load_state_from_group(v) for k, v in group.items() if isinstance(v, h5py.Group)},
    )


def load_state_from_file(f: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> LudwigModuleState:
    """Loads Ludwig Module state from a file."""
    with h5py.File(f, "r") as f:
        # The file object does double duty as the HDF5 root group
        return _load_state_from_group(f)


def save(object: LudwigModule, f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    """Saves Ludwig object to file or buffer."""
    object_state = object.get_state()
    save_state_to_file(object_state, f)


def load(f: Union[str, os.PathLike, BinaryIO, IO[bytes]], device: str = None) -> LudwigModule:
    """Loads saved Ludwig module from file or buffer. If the module has parameters which are torch Tensors, device
    specifies the device where tensors will be instantiated.

    Args:
        f: The file path or object to load from.
        device: 'cuda' or 'cpu'
    """
    pass


def update(object_state: LudwigModuleState) -> LudwigModuleState:
    """Update saved Ludwig object from previous version."""
    if object_state == LUDWIG_VERSION:
        return object_state
    else:
        # TODO: check version, apply migrations if needed.
        object_state.ludwig_version = LUDWIG_VERSION
        return object_state
