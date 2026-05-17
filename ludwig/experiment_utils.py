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
"""Utilities for generating experiment metadata and compute descriptions.

These utilities are used by ``LudwigModel`` and by the experiment CLI, and
are kept here to avoid coupling the CLI (``experiment.py``) to the full
``api.py`` module.
"""

import sys
from collections import OrderedDict

import pandas as pd
import torch

from ludwig.api_annotations import PublicAPI
from ludwig.backend import Backend
from ludwig.globals import LUDWIG_VERSION
from ludwig.types import ModelConfigDict, TrainingSetMetadataDict
from ludwig.utils.data_utils import figure_data_format
from ludwig.utils.misc_utils import get_commit_hash


def _get_compute_description(backend: Backend) -> dict:
    """Returns the compute description for the backend."""
    compute_description = {"num_nodes": backend.num_nodes}

    if torch.cuda.is_available():
        # Assumption: All nodes are of the same instance type (not yet verified across Ray workers).
        compute_description.update(
            {
                "gpus_per_node": torch.cuda.device_count(),
                "arch_list": torch.cuda.get_arch_list(),
                "gencode_flags": torch.cuda.get_gencode_flags(),
                "devices": {},
            }
        )
        for i in range(torch.cuda.device_count()):
            compute_description["devices"][i] = {
                "gpu_type": torch.cuda.get_device_name(i),
                "device_capability": torch.cuda.get_device_capability(i),
                "device_properties": str(torch.cuda.get_device_properties(i)),
            }

    return compute_description


@PublicAPI
def get_experiment_description(
    config: ModelConfigDict,
    dataset: str | dict | pd.DataFrame | None = None,
    training_set: str | dict | pd.DataFrame | None = None,
    validation_set: str | dict | pd.DataFrame | None = None,
    test_set: str | dict | pd.DataFrame | None = None,
    training_set_metadata: TrainingSetMetadataDict | None = None,
    data_format: str | None = None,
    backend: Backend | None = None,
    random_seed: int | None = None,
) -> dict:
    description = OrderedDict()
    description["ludwig_version"] = LUDWIG_VERSION
    description["command"] = " ".join(sys.argv)

    commit_hash = get_commit_hash()
    if commit_hash is not None:
        description["commit_hash"] = commit_hash[:12]

    if random_seed is not None:
        description["random_seed"] = random_seed

    if isinstance(dataset, str):
        description["dataset"] = dataset
    if isinstance(training_set, str):
        description["training_set"] = training_set
    if isinstance(validation_set, str):
        description["validation_set"] = validation_set
    if isinstance(test_set, str):
        description["test_set"] = test_set
    if training_set_metadata is not None:
        description["training_set_metadata"] = training_set_metadata

    # determine data format if not provided or auto
    if not data_format or data_format == "auto":
        data_format = figure_data_format(dataset, training_set, validation_set, test_set)

    if data_format:
        description["data_format"] = str(data_format)

    description["config"] = config
    description["torch_version"] = torch.__version__
    description["compute"] = _get_compute_description(backend)

    return description
