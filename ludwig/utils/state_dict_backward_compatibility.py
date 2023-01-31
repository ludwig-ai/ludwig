# Copyright (c) 2023 Predibase, Inc.
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


def _update_transformers_to_freeze_module(state_dict):
    """Updates pre-trained encoders which were saved prior to the addition of FreezeModule."""
    return {
        k.replace("encoder_obj.transformer.", "encoder_obj.transformer.module.")
        if "encoder_obj.transformer.module" not in k
        else k: v
        for k, v in state_dict.items()
    }


def update_state_dict(state_dict):
    """Checks state_dict on load, updates state dict if needed."""
    state_dict = _update_transformers_to_freeze_module(state_dict)
    return state_dict
