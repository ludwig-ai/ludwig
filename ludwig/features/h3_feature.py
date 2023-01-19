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
import logging
from typing import List

import numpy as np
import torch

from ludwig.constants import COLUMN, H3, PROC_COLUMN
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature
from ludwig.schema.features.h3_feature import H3InputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.h3_util import h3_to_components
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)

MAX_H3_RESOLUTION = 15
H3_VECTOR_LENGTH = MAX_H3_RESOLUTION + 4
H3_PADDING_VALUE = 7


class _H3Preprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.max_h3_resolution = MAX_H3_RESOLUTION
        self.h3_padding_value = H3_PADDING_VALUE
        self.computed_fill_value = float(metadata["preprocessing"]["computed_fill_value"])

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if torch.jit.isinstance(v, List[torch.Tensor]):
            v = torch.stack(v)

        if not torch.jit.isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")

        v = torch.nan_to_num(v, nan=self.computed_fill_value)
        v = v.long()

        outputs: List[torch.Tensor] = []
        for v_i in v:
            components = h3_to_components(v_i)
            header: List[int] = [
                components.mode,
                components.edge,
                components.resolution,
                components.base_cell,
            ]
            cells_padding: List[int] = [self.h3_padding_value] * (self.max_h3_resolution - len(components.cells))
            output = torch.tensor(header + components.cells + cells_padding, dtype=torch.uint8, device=v.device)
            outputs.append(output)

        return torch.stack(outputs)


class H3FeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return H3

    @staticmethod
    def cast_column(column, backend):
        try:
            return column.astype(int)
        except ValueError:
            logger.warning("H3Feature could not be read as int directly. Reading as float and converting to int.")
            return column.astype(float).astype(int)

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        return {}

    @staticmethod
    def h3_to_list(h3_int):
        components = h3_to_components(h3_int)
        header = [components.mode, components.edge, components.resolution, components.base_cell]
        cells_padding = [H3_PADDING_VALUE] * (MAX_H3_RESOLUTION - len(components.cells))
        return header + components.cells + cells_padding

    @staticmethod
    def add_feature_data(
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        column = input_df[feature_config[COLUMN]]
        if column.dtype == object:
            column = backend.df_engine.map_objects(column, int)
        column = backend.df_engine.map_objects(column, H3FeatureMixin.h3_to_list)

        proc_df[feature_config[PROC_COLUMN]] = backend.df_engine.map_objects(
            column, lambda x: np.array(x, dtype=np.uint8)
        )
        return proc_df


class H3InputFeature(H3FeatureMixin, InputFeature):
    def __init__(self, input_feature_config: H3InputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.uint8, torch.int64]
        assert len(inputs.shape) == 2

        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    @property
    def input_dtype(self):
        return torch.uint8

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _H3Preprocessing(metadata)

    @staticmethod
    def get_schema_cls():
        return H3InputFeatureConfig
