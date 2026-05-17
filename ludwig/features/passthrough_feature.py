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
"""Passthrough input feature for embed-mode inference.

When encoder embeddings are pre-cached during preprocessing, the ECD model
replaces the real input feature with a PassthroughInputFeature so that
``forward()`` simply returns the already-embedded tensor without re-encoding.
"""

import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.features.base_feature import InputFeature
from ludwig.schema.features.base import BaseFeatureConfig
from ludwig.types import TrainingSetMetadataDict
from ludwig.utils.types import PreprocessingInput


class PassthroughPreprocModule(torch.nn.Module):
    """Combines preprocessing and encoding into a single module for TorchScript inference.

    For encoder outputs that were cached during preprocessing, the encoder is simply the identity function in the ECD
    module. As such, we need this module to apply the encoding that would normally be done during preprocessing for
    realtime inference.
    """

    def __init__(self, preproc: torch.nn.Module, encoder: torch.nn.Module):
        super().__init__()
        self.preproc = preproc
        self.encoder = encoder

    def forward(self, v: PreprocessingInput) -> torch.Tensor:
        preproc_v = self.preproc(v)
        return self.encoder(preproc_v)


class PassthroughInputFeature(InputFeature):
    """A transparent identity-function wrapper around an input feature whose encoder was pre-cached.

    Used when encoder embeddings were computed during preprocessing (embed mode). The passthrough
    delegates shape/type queries to the wrapped feature's encoder so the rest of the model sees a
    consistent interface, while ``forward()`` simply returns the already-embedded tensor unchanged.
    """

    def __init__(self, config: BaseFeatureConfig, wrapped: InputFeature):
        super().__init__(config)
        self._wrapped = wrapped

    def forward(self, inputs, mask=None) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"PassthroughInputFeature forward expects a torch.Tensor, got {type(inputs).__name__}.")
        return {ENCODER_OUTPUT: inputs}

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def input_shape(self) -> torch.Size:
        return self._wrapped.encoder_obj.output_shape

    @property
    def output_shape(self) -> torch.Size:
        return self._wrapped.encoder_obj.output_shape

    def update_config_with_metadata(self, feature_config, feature_metadata, *args, **kwargs) -> None:
        return self._wrapped.update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs)

    def get_schema_cls(self) -> type:
        return self._wrapped.get_schema_cls()

    def create_preproc_module(self, metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return PassthroughPreprocModule(self._wrapped.create_preproc_module(metadata), self._wrapped)

    def type(self) -> str:
        return self._wrapped.type()

    def unskip(self) -> InputFeature:
        return self._wrapped

    @property
    def encoder_obj(self) -> torch.nn.Module:
        return self._wrapped.encoder_obj


def create_passthrough_input_feature(feature: InputFeature, config: BaseFeatureConfig) -> PassthroughInputFeature:
    """Wraps *feature* in a :class:`PassthroughInputFeature` shim.

    The shim acts as a transparent identity function — useful when encoder embeddings
    were cached during preprocessing and the model should skip re-encoding.
    """
    return PassthroughInputFeature(config, wrapped=feature)
