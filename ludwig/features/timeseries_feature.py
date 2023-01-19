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
from typing import List

import numpy as np
import torch

from ludwig.constants import COLUMN, NAME, PROC_COLUMN, TIMESERIES
from ludwig.features.base_feature import BaseFeatureMixin
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.schema.features.timeseries_feature import TimeseriesInputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.tokenizers import get_tokenizer_from_registry, TORCHSCRIPT_COMPATIBLE_TOKENIZERS
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


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

        ts_vectors = backend.df_engine.map_objects(timeseries, lambda ts: np.array(tokenizer(ts)).astype(np.float32))

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
        timeseries_data = TimeseriesFeatureMixin.build_matrix(
            column,
            preprocessing_parameters["tokenizer"],
            metadata["max_timeseries_length"],
            preprocessing_parameters["padding_value"],
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
        return torch.Size([self.encoder_obj.config.max_sequence_length])

    @property
    def input_dtype(self):
        return torch.float32

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.max_sequence_length = feature_metadata["max_timeseries_length"]

    @staticmethod
    def get_schema_cls():
        return TimeseriesInputFeatureConfig

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _TimeseriesPreprocessing(metadata)


# this is still WIP
# class TimeseriesOutputFeature(TimeseriesBaseFeature, SequenceOutputFeature):
#     def __init__(self, feature):
#         super().__init__(feature)
#         self.decoder = 'generator'
#
#         self.loss = {
#             'weight': 1,
#             TYPE: 'softmax_cross_entropy',
#             'class_weights': 1,
#             'class_similarities_temperature': 0
#         }
#         self.num_classes = 0
##
#         self.decoder_obj = self.get_sequence_decoder(feature)
#
#     def _get_output_placeholder(self):
#         return tf.placeholder(
#             tf.float32,
#             [None, self.max_sequence_length],
#             name='{}_placeholder'.format(self.feature_name)
#         )
#
#     def _get_metrics(self, targets, predictions):
#         with tf.variable_scope('metrics_{}'.format(self.feature_name)):
#             error_val = error(targets, predictions, self.feature_name)
#             absolute_error_val = absolute_error(targets, predictions, self.feature_name)
#             squared_error_val = squared_error(targets, predictions, self.feature_name)
#             r2_val = r2(targets, predictions, self.feature_name)
#         return error_val, squared_error_val, absolute_error_val, r2_val
#
#     def _get_loss(self, targets, predictions):
#         with tf.variable_scope('loss_{}'.format(self.feature_name)):
#             if self.loss[TYPE] == 'mean_squared_error':
#                 train_loss = tf.losses.mean_squared_error(
#                     labels=targets,
#                     predictions=predictions,
#                     reduction=Reduction.NONE
#                 )
#             elif self.loss[TYPE] == 'mean_absolute_error':
#                 train_loss = tf.losses.absolute_difference(
#                     labels=targets,
#                     predictions=predictions,
#                     reduction=Reduction.NONE
#                 )
#             else:
#                 train_loss = None
#                 train_mean_loss = None
#                 raise ValueError(
#                     'Unsupported loss type {}'.format(self.loss[TYPE])
#                 )
#             train_mean_loss = tf.reduce_mean(
#                 train_loss,
#                 name='train_mean_loss_{}'.format(self.feature_name)
#             )
#         return train_mean_loss, train_loss
#
#     def build_output(
#             self,
#             hidden,
#             hidden_size,
#             dropout=None,
#             is_training=None,
#             **kwargs
#     ):
#         output_tensors = {}
#
#         # ================ Placeholder ================
#         targets = self._get_output_placeholder()
#         output_tensors[self.feature_name] = targets
#         logger.debug('  targets_placeholder: {0}'.format(targets))
#
#         # ================ Predictions ================
#         (
#             predictions_sequence,
#             predictions_sequence_scores,
#             predictions_sequence_length,
#             last_predictions,
#             targets_sequence_length,
#             last_targets,
#             eval_logits,
#             train_logits,
#             class_weights,
#             class_biases
#         ) = self.sequence_predictions(
#             targets,
#             self.decoder_obj,
#             hidden,
#             hidden_size,
#             is_timeseries=True
#         )
#
#         output_tensors[LAST_PREDICTIONS + '_' + self.feature_name] = last_predictions
#         output_tensors[PREDICTIONS + '_' + self.feature_name] = predictions_sequence
#         output_tensors[LENGTHS + '_' + self.feature_name] = predictions_sequence_length
#
#         # ================ metrics ================
#         (
#             error_val,
#             squared_error_val,
#             absolute_error_val,
#             r2_val
#         ) = self._get_metrics(
#             targets,
#             predictions_sequence
#         )
#
#         output_tensors[ERROR + '_' + self.feature_name] = error_val
#         output_tensors[SQUARED_ERROR + '_' + self.feature_name] = squared_error_val
#         output_tensors[ABSOLUTE_ERROR + '_' + self.feature_name] = absolute_error_val
#         output_tensors[R2 + '_' + self.feature_name] = r2_val
#
#         if 'sampled' not in self.loss[TYPE]:
#             tf.summary.scalar(
#                 'batch_train_mean_squared_error_{}'.format(self.feature_name),
#                 tf.reduce_mean(squared_error)
#             )
#             tf.summary.scalar(
#                 'batch_train_mean_absolute_error_{}'.format(self.feature_name),
#                 tf.reduce_mean(absolute_error)
#             )
#             tf.summary.scalar(
#                 'batch_train_mean_r2_{}'.format(self.feature_name),
#                 tf.reduce_mean(r2)
#             )
#
#         # ================ Loss ================
#         train_mean_loss, eval_loss = self._get_loss(
#             targets,
#             predictions_sequence
#         )
#
#         output_tensors[TRAIN_MEAN_LOSS + '_' + self.feature_name] = train_mean_loss
#         output_tensors[EVAL_LOSS + '_' + self.feature_name] = eval_loss
#
#         tf.summary.scalar(
#             'batch_train_mean_loss_{}'.format(self.feature_name),
#             train_mean_loss,
#         )
#
#         return train_mean_loss, eval_loss, output_tensors
#
#     default_validation_metric = LOSS
#
#     output_config = OrderedDict([
#         (LOSS, {
#             'output': EVAL_LOSS,
#             'aggregation': SUM,
#             'value': 0,
#             TYPE: METRIC
#         }),
#         (MEAN_SQUARED_ERROR, {
#             'output': SQUARED_ERROR,
#             'aggregation': SUM,
#             'value': 0,
#             TYPE: METRIC
#         }),
#         (MEAN_ABSOLUTE_ERROR, {
#             'output': ABSOLUTE_ERROR,
#             'aggregation': SUM,
#             'value': 0,
#             TYPE: METRIC
#         }),
#         (R2, {
#             'output': R2,
#             'aggregation': SUM,
#             'value': 0,
#             TYPE: METRIC
#         }),
#         (ERROR, {
#             'output': ERROR,
#             'aggregation': SUM,
#             'value': 0,
#             TYPE: METRIC
#         }),
#         (PREDICTIONS, {
#             'output': PREDICTIONS,
#             'aggregation': APPEND,
#             'value': [],
#             TYPE: PREDICTION
#         }),
#         (LENGTHS, {
#             'output': LENGTHS,
#             'aggregation': APPEND,
#             'value': [],
#             TYPE: PREDICTION
#         })
#     ])
#
#     @classmethod
#     def get_output_dtype(cls):
#         return tf.float32
#
#     def get_output_shape(self):
#         return self.max_sequence_length,
#
#
#     @staticmethod
#     def update_config_with_metadata(
#             output_feature,
#             feature_metadata,
#             *args,
#             **kwargs
#     ):
#         output_feature['max_sequence_length'] = feature_metadata[
#             'max_timeseries_length'
#         ]
#
#     @staticmethod
#     def calculate_overall_stats(
#             test_stats,
#             output_feature,
#             dataset,
#             train_set_metadata
#     ):
#         pass
#
#
#     def postprocess_predictions(
#             self,
#             result,
#             metadata,
#             output_directory,
#             skip_save_unprocessed_output=False,
#     ):
#         pass
#
#     @staticmethod
#     def populate_defaults(output_feature):
#         set_default_value(
#             output_feature,
#             LOSS,
#             {TYPE: 'mean_absolute_error', 'weight': 1}
#         )
#         set_default_value(output_feature[LOSS], TYPE, 'mean_absolute_error')
#         set_default_value(output_feature[LOSS], 'weight', 1)
#
#         set_default_value(output_feature, 'decoder', 'generator')
#
#         if output_feature['decoder'] == 'generator':
#             set_default_value(output_feature, 'cell_type', 'rnn')
#             set_default_value(output_feature, 'state_size', 256)
#             set_default_value(output_feature, 'embedding_size', 1)
#             set_default_value(output_feature, 'attention_mechanism', None)
#             if output_feature['attention_mechanism'] is not None:
#                 set_default_value(output_feature, 'reduce_input', None)
#             set_default_value(output_feature, 'decoder', 'generator')
#             set_default_value(output_feature, 'decoder', 'generator')
#             set_default_value(output_feature, 'decoder', 'generator')
#             set_default_value(output_feature, 'decoder', 'generator')
#
#         if output_feature['decoder'] == 'tagger':
#             if 'reduce_input' not in output_feature:
#                 output_feature['reduce_input'] = None
#
#         set_default_value(output_feature, 'dependencies', [])
#         set_default_value(output_feature, 'reduce_input', SUM)
#         set_default_value(output_feature, 'reduce_dependencies', SUM)
