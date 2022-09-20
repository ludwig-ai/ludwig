#! /usr/bin/env python
# Copyright (c) 2021 Linux Foundation.
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

import warnings

import psutil


def get_num_cpus() -> int:
    """Get num CPUs, accounting for hyperthreading."""
    # Count of logical CPUs, i.e., cores with hyper-threading
    cpu_count = psutil.cpu_count()
    return cpu_count if cpu_count is not None else 1


def get_num_gpus() -> int:
    try:
        import GPUtil

        if GPUtil.getGPUs():
            return len(GPUtil.getGPUs())
    except Exception as e:
        warnings.warn(f"GPUtil is not installed. Assuming no GPUs are available. {e}")
    return 0
