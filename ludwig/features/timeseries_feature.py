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
from typing import Any, Dict, List, Union

import numpy as np
import torch

from ludwig.constants import (
    COLUMN,
    FILL_WITH_CONST,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NAME,
    PROC_COLUMN,
    TIED,
    TIMESERIES,
)
from ludwig.features.base_feature import BaseFeatureMixin
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.utils.misc_utils import get_from_registry, set_default_values
from ludwig.utils.strings_utils import tokenizer_registry
from ludwig.utils.tokenizers import TORCHSCRIPT_COMPATIBLE_TOKENIZERS

logger = logging.getLogger(__name__)


class _TimeseriesPreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by TimeseriesFeatureMixin.add_feature_data."""

    def __init__(self, metadata: Dict[str, Any]):
        super().__init__()
        if metadata["preprocessing"]["tokenizer"] not in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
            raise ValueError(
                f"{metadata['preprocessing']['tokenizer']} is not supported by torchscript. Please use "
                f"one of {TORCHSCRIPT_COMPATIBLE_TOKENIZERS}."
            )
        self.tokenizer = get_from_registry(metadata["preprocessing"]["tokenizer"], tokenizer_registry)()
        self.padding = metadata["preprocessing"]["padding"]
        self.padding_value = metadata["preprocessing"]["padding_value"]
        self.max_timeseries_length = int(metadata["max_timeseries_length"])

    def forward(self, v: Union[List[str], List[torch.Tensor], torch.Tensor]):
        """Takes a list of strings and returns a tensor of token ids."""
        if not torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        sequences = self.tokenizer(v)
        # refines type of sequences from Any to List[List[str]]
        assert torch.jit.isinstance(sequences, List[List[str]]), "sequences is not a list of lists."

        float_sequences: List[List[float]] = [[float(s) for s in sequence] for sequence in sequences]
        timeseries_matrix = torch.full(
            [len(float_sequences), self.max_timeseries_length], self.padding_value, dtype=torch.float32
        )
        for sample_idx, float_sequence in enumerate(float_sequences):
            limit = min(len(float_sequence), self.max_timeseries_length)
            if self.padding == "right":
                timeseries_matrix[sample_idx][:limit] = torch.tensor(float_sequence[:limit])
            else:  # if self.padding == 'left
                timeseries_matrix[sample_idx][self.max_timeseries_length - limit :] = torch.tensor(
                    float_sequence[:limit]
                )
        return timeseries_matrix


class TimeseriesFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return TIMESERIES

    @staticmethod
    def preprocessing_defaults():
        return {
            "timeseries_length_limit": 256,
            "padding_value": 0,
            "padding": "right",
            "tokenizer": "space",
            "missing_value_strategy": FILL_WITH_CONST,
            "fill_value": "",
        }

    @staticmethod
    def preprocessing_schema():
        return {
            "timeseries_length_limit": {"type": "integer", "minimum": 0},
            "padding_value": {"type": "number"},
            "padding": {"type": "string", "enum": ["right", "left"]},
            "tokenizer": {"type": "string", "enum": sorted(list(tokenizer_registry.keys()))},
            "missing_value_strategy": {"type": "string", "enum": MISSING_VALUE_STRATEGY_OPTIONS},
            "fill_value": {"type": "string"},
            "computed_fill_value": {"type": "string"},
        }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        column = column.astype(str)
        tokenizer = get_from_registry(preprocessing_parameters["tokenizer"], tokenizer_registry)()
        max_length = 0
        for timeseries in column:
            processed_line = tokenizer(timeseries)
            max_length = max(max_length, len(processed_line))
        max_length = min(preprocessing_parameters["timeseries_length_limit"], max_length)

        return {"max_timeseries_length": max_length}

    @staticmethod
    def build_matrix(timeseries, tokenizer_name, length_limit, padding_value, padding, backend):
        tokenizer = get_from_registry(tokenizer_name, tokenizer_registry)()

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
    def feature_data(column, metadata, preprocessing_parameters, backend):
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
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        proc_df[feature_config[PROC_COLUMN]] = TimeseriesFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]].astype(str),
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


class TimeseriesInputFeature(TimeseriesFeatureMixin, SequenceInputFeature):
    encoder = "parallel_cnn"
    max_sequence_length = None

    def __init__(self, feature, encoder_obj=None):
        # add required sequence encoder parameters for time series
        feature["embedding_size"] = 1
        feature["should_embed"] = False

        # initialize encoder for time series
        super().__init__(feature, encoder_obj=encoder_obj)

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.float16, torch.float32, torch.float64]
        assert len(inputs.shape) == 2

        inputs_exp = inputs.type(torch.float32)
        encoder_output = self.encoder_obj(inputs_exp, mask=mask)

        return encoder_output

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def input_dtype(self):
        return torch.float32

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        input_feature["max_sequence_length"] = feature_metadata["max_timeseries_length"]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                TIED: None,
                "encoder": "parallel_cnn",
            },
        )

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
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
#
#         self.overwrite_defaults(feature)
#
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
