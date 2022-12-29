# Copyright (c) 2020 Uber Technologies, Inc.
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
import os
from typing import Any, List, Optional

import torch

try:
    import horovod.torch

    _HVD = horovod.torch
except (ModuleNotFoundError, ImportError):
    _HVD = None


def initialize_horovod():
    if not _HVD:
        raise ValueError(
            "Horovod backend specified, "
            "but cannot import `horovod.torch`. "
            "Install Horovod following the instructions at: "
            "https://github.com/horovod/horovod"
        )
    _HVD.init()
    return _HVD


def has_horovodrun():
    """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
    return "OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ


def gather_all_tensors(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:
    """Function to gather all tensors from several processes onto a list that is broadcast to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    :param result: the value to sync
    :param group: the process group to gather results from (not supported: always uses world)

    :return: list with size equal to the process group where gathered_result[i]
             corresponds to result tensor from process i
    """
    if group is not None:
        raise ValueError("Horovod does not support allgather using a subcommunicator at this time. " "Unset `group`.")

    if _HVD is None or not _HVD.is_initialized():
        return [result]

    if len(result.shape) == 0:
        # Convert scalars to single dimension tensors
        result = result.reshape(1)

    is_bool = False
    if result.dtype == torch.bool:
        # need to convert to int due to Horovod limitation
        result = result.int()
        is_bool = True

    # Add extra dimension to the tensors to be gathered
    result = result.unsqueeze(0)

    # sync and gather all
    gathered = _HVD.allgather(result)
    gathered_result = list(gathered.split(1, dim=0))

    if is_bool:
        # convert back if needed
        gathered_result = [t.bool() for t in gathered_result]

    return gathered_result


def is_distributed_available() -> bool:
    return _HVD is not None and (_HVD.is_initialized() or os.environ.get("HOROVOD_RANK"))
