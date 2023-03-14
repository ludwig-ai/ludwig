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
import logging
from typing import Dict, List, TYPE_CHECKING, Union

import numpy as np
import torch

from ludwig.constants import COLUMN, HIDDEN, LOGITS, NAME, PREDICTIONS, PROC_COLUMN, TIMESERIES
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature, PredictModule
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.vector_feature import _VectorPostprocessing, _VectorPredict
from ludwig.schema.features.timeseries_feature import TimeseriesInputFeatureConfig, TimeseriesOutputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.tokenizers import get_tokenizer_from_registry, TORCHSCRIPT_COMPATIBLE_TOKENIZERS
from ludwig.utils.types import Series, TorchscriptPreprocessingInput

if TYPE_CHECKING:
    from ludwig.backend.base import Backend

logger = logging.getLogger(__name__)


def create_time_delay_embedding(
    series: Series, window_size: int, horizon: int, padding_value: int, backend: "Backend"
) -> Series:
    """Time delay embedding from:

    https://towardsdatascience.com/machine-learning-for-forecasting-transformations-and-feature-extraction-bbbea9de0ac2

    Args:
        series: Column-major timeseries data.
        window_size: Size of the lookback sliding window for timeseries inputs.
        horizon: Size of the forward-looking horizon for timeseries outputs.
        padding_value: Value to pad out the window when there is not enough data around the observation.

    Returns:
        A column of timeseries window arrays in row-major format for training.
    """
    # Replace default fill value of "" with nan as we will be assuming numeric values here
    series = series.replace("", np.nan)

    # Create the list of shifts we want to perform over the series.
    # For backwards looking shifts, we want to include the current element, while for forward looking shifts we do not.
    # Example:
    #   window_size=3, horizon=0 --> shift_offsets=[2, 1, 0]
    #   window_size=0, horizon=2 --> shift_offsets=[-1, -2]
    shift_offsets = list(range(window_size - 1, -(horizon + 1), -1))
    shifts = [series.shift(i) for i in shift_offsets]
    df = backend.df_engine.df_lib.concat(shifts, axis=1)
    df.columns = [f"__tmp_column_{j}" for j in shift_offsets]
    return df.apply(lambda x: np.nan_to_num(np.array(x.tolist()).astype(np.float32), nan=padding_value), axis=1)


class _TimeseriesPreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by TimeseriesFeatureMixin.add_feature_data."""

    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        if metadata["preprocessing"]["tokenizer"] not in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
            raise ValueError(
                f"{metadata['preprocessing']['tokenizer']} is not supported by torchscript. Please use "
                f"one of {TORCHSCRIPT_COMPATIBLE_TOKENIZERS}."
            )
        self.tokenizer = get_tokenizer_from_registry(metadata["preprocessing"]["tokenizer"])()
        self.padding = metadata["preprocessing"]["padding"]
        self.padding_value = float(metadata["preprocessing"]["padding_value"])
        self.max_timeseries_length = int(metadata["max_timeseries_length"])
        self.computed_fill_value = metadata["preprocessing"]["computed_fill_value"]

    def _process_str_sequence(self, sequence: List[str], limit: int) -> torch.Tensor:
        float_sequence = [float(s) for s in sequence[:limit]]
        return torch.tensor(float_sequence)

    def _nan_to_fill_value(self, v: torch.Tensor) -> torch.Tensor:
        if v.isnan().any():
            tokenized_fill_value = self.tokenizer(self.computed_fill_value)
            # refines type of sequences from Any to List[str]
            assert torch.jit.isinstance(tokenized_fill_value, List[str])
            return self._process_str_sequence(tokenized_fill_value, self.max_timeseries_length)
        return v

    def forward_list_of_tensors(self, v: List[torch.Tensor]) -> torch.Tensor:
        v = [self._nan_to_fill_value(v_i) for v_i in v]

        if self.padding == "right":
            timeseries_matrix = torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=self.padding_value)
            timeseries_matrix = timeseries_matrix[:, : self.max_timeseries_length]
        else:
            reversed_timeseries = [torch.flip(v_i[: self.max_timeseries_length], dims=(0,)) for v_i in v]
            reversed_timeseries_padded = torch.nn.utils.rnn.pad_sequence(
                reversed_timeseries, batch_first=True, padding_value=self.padding_value
            )
            timeseries_matrix = torch.flip(reversed_timeseries_padded, dims=(1,))
        return timeseries_matrix

    def forward_list_of_strs(self, v: List[str]) -> torch.Tensor:
        v = [self.computed_fill_value if s == "nan" else s for s in v]

        sequences = self.tokenizer(v)
        # refines type of sequences from Any to List[List[str]]
        assert torch.jit.isinstance(sequences, List[List[str]]), "sequences is not a list of lists."

        timeseries_matrix = torch.full(
            [len(sequences), self.max_timeseries_length], self.padding_value, dtype=torch.float32
        )
        for sample_idx, str_sequence in enumerate(sequences):
            limit = min(len(str_sequence), self.max_timeseries_length)
            float_sequence = self._process_str_sequence(str_sequence, limit)
            if self.padding == "right":
                timeseries_matrix[sample_idx][:limit] = float_sequence
            else:  # if self.padding == 'left
                timeseries_matrix[sample_idx][self.max_timeseries_length - limit :] = float_sequence
        return timeseries_matrix

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        """Takes a list of float values and creates a padded torch.Tensor."""
        if torch.jit.isinstance(v, List[torch.Tensor]):
            return self.forward_list_of_tensors(v)
        if torch.jit.isinstance(v, List[str]):
            return self.forward_list_of_strs(v)
        raise ValueError(f"Unsupported input: {v}")


class TimeseriesFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return TIMESERIES

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        window_size = preprocessing_parameters.get("window_size", 0) or preprocessing_parameters.get("horizon", 0)
        if window_size > 0:
            # Column-major data
            return {"max_timeseries_length": window_size}

        column = column.astype(str)
        tokenizer = get_tokenizer_from_registry(preprocessing_parameters["tokenizer"])()
        max_length = 0
        for timeseries in column:
            processed_line = tokenizer(timeseries)
            max_length = max(max_length, len(processed_line))
        max_length = min(preprocessing_parameters["timeseries_length_limit"], max_length)

        return {"max_timeseries_length": max_length}

    @staticmethod
    def build_matrix(timeseries, tokenizer_name, length_limit, padding_value, padding, backend):
        tokenizer = get_tokenizer_from_registry(tokenizer_name)()

        ts_vectors = backend.df_engine.map_objects(
            timeseries, lambda ts: np.nan_to_num(np.array(tokenizer(ts)).astype(np.float32), nan=padding_value)
        )

        max_length = backend.df_engine.compute(ts_vectors.map(len).max())
        if max_length < length_limit:
            logger.debug(f"max length of {tokenizer_name}: {max_length} < limit: {length_limit}")
        max_length = length_limit

        def pad(vector):
            padded = np.full((max_length,), padding_value, dtype=np.float32)
            limit = min(vector.shape[0], max_length)
            if padding == "right":
                padded[:limit] = vector[:limit]
            else:  # if padding == 'left
                padded[max_length - limit :] = vector[:limit]
            return padded

        return backend.df_engine.map_objects(ts_vectors, pad)

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters: PreprocessingConfigDict, backend):
        padding_value = preprocessing_parameters["padding_value"]

        window_size = preprocessing_parameters.get("window_size", 0)
        horizon = preprocessing_parameters.get("horizon", 0)
        if window_size > 0 or horizon > 0:
            # Column-major data. Convert the column into the row-major embedding
            return create_time_delay_embedding(column, window_size, horizon, padding_value, backend)

        timeseries_data = TimeseriesFeatureMixin.build_matrix(
            column,
            preprocessing_parameters["tokenizer"],
            metadata["max_timeseries_length"],
            padding_value,
            preprocessing_parameters["padding"],
            backend,
        )
        return timeseries_data

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
        proc_df[feature_config[PROC_COLUMN]] = TimeseriesFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]].astype(str),
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


class TimeseriesInputFeature(TimeseriesFeatureMixin, SequenceInputFeature):
    def __init__(self, input_feature_config: TimeseriesInputFeatureConfig, encoder_obj=None, **kwargs):
        # add required sequence encoder parameters for time series
        input_feature_config.encoder.embedding_size = 1
        input_feature_config.encoder.should_embed = False

        # SequenceInputFeauture's constructor initializes the encoder.
        super().__init__(input_feature_config, encoder_obj=encoder_obj, **kwargs)

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.float16, torch.float32, torch.float64]
        assert len(inputs.shape) == 2

        inputs_exp = inputs.type(torch.float32)
        encoder_output = self.encoder_obj(inputs_exp, mask=mask)

        return encoder_output

    @property
    def input_shape(self) -> torch.Size:
        return self.encoder_obj.input_shape

    @property
    def input_dtype(self):
        return torch.float32

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.input_size = feature_metadata["max_timeseries_length"]
        feature_config.encoder.max_sequence_length = feature_metadata["max_timeseries_length"]

    @staticmethod
    def get_schema_cls():
        return TimeseriesInputFeatureConfig

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _TimeseriesPreprocessing(metadata)


class TimeseriesOutputFeature(TimeseriesFeatureMixin, OutputFeature):
    def __init__(
        self,
        output_feature_config: Union[TimeseriesOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        self.horizon = output_feature_config.horizon
        super().__init__(output_feature_config, output_features, **kwargs)
        output_feature_config.decoder.output_size = self.horizon

        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def loss_kwargs(self):
        return self.loss.to_dict()

    def metric_kwargs(self):
        return dict(num_outputs=self.output_shape[0])

    def create_predict_module(self) -> PredictModule:
        return _VectorPredict()

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.horizon])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.horizon = feature_metadata["max_timeseries_length"]

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
        self,
        result,
        metadata,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            result[predictions_col] = result[predictions_col].map(lambda pred: pred.tolist())
        return result

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _VectorPostprocessing()

    @staticmethod
    def get_schema_cls():
        return TimeseriesOutputFeatureConfig
