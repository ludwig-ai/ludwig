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
from functools import partial
from typing import Any, Dict

import torch

from ludwig.constants import (
    COLUMN,
    DECODER,
    EDIT_DISTANCE,
    ENCODER,
    FILL_WITH_CONST,
    LAST_ACCURACY,
    LAST_PREDICTIONS,
    LENGTHS,
    LOSS,
    NAME,
    PERPLEXITY,
    PREDICTIONS,
    PROBABILITIES,
    PROBABILITY,
    PROC_COLUMN,
    TEXT,
    TIED,
    TOKEN_ACCURACY,
    TYPE,
)
from ludwig.encoders.registry import get_encoder_cls
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature
from ludwig.features.feature_utils import compute_sequence_probability, compute_token_probabilities
from ludwig.features.sequence_feature import (
    _SequencePostprocessing,
    _SequencePreprocessing,
    SequenceInputFeature,
    SequenceOutputFeature,
)
from ludwig.schema.features.text_feature import TextInputFeatureConfig, TextOutputFeatureConfig
from ludwig.schema.features.utils import register_input_feature, register_output_feature
from ludwig.utils.math_utils import softmax
from ludwig.utils.misc_utils import set_default_values
from ludwig.utils.strings_utils import (
    build_sequence_matrix,
    create_vocabulary,
    PADDING_SYMBOL,
    SpecialSymbol,
    UNKNOWN_SYMBOL,
)
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)


class TextFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return TEXT

    @staticmethod
    def preprocessing_defaults():
        return {
            "tokenizer": "space_punct",
            "pretrained_model_name_or_path": None,
            "vocab_file": None,
            "max_sequence_length": 256,
            "most_common": 20000,
            "padding_symbol": PADDING_SYMBOL,
            "unknown_symbol": UNKNOWN_SYMBOL,
            "padding": "right",
            "lowercase": True,
            "missing_value_strategy": FILL_WITH_CONST,
            "fill_value": UNKNOWN_SYMBOL,
        }

    @staticmethod
    def cast_column(column, backend):
        return column.astype(str)

    @staticmethod
    def feature_meta(column, preprocessing_parameters, backend):
        (
            idx2str,
            str2idx,
            str2freq,
            max_len,
            max_len_99ptile,
            pad_idx,
            padding_symbol,
            unknown_symbol,
        ) = create_vocabulary(
            column,
            tokenizer_type=preprocessing_parameters["tokenizer"],
            num_most_frequent=preprocessing_parameters["most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            vocab_file=preprocessing_parameters["vocab_file"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )
        return (
            idx2str,
            str2idx,
            str2freq,
            max_len,
            max_len_99ptile,
            pad_idx,
            padding_symbol,
            unknown_symbol,
        )

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        tf_meta = TextFeatureMixin.feature_meta(column, preprocessing_parameters, backend)
        (
            idx2str,
            str2idx,
            str2freq,
            max_len,
            max_len_99ptile,
            pad_idx,
            padding_symbol,
            unknown_symbol,
        ) = tf_meta
        max_len = min(preprocessing_parameters["max_sequence_length"], max_len)
        max_len_99ptile = min(max_len, max_len_99ptile)
        return {
            "idx2str": idx2str,
            "str2idx": str2idx,
            "str2freq": str2freq,
            "vocab_size": len(idx2str),
            "max_sequence_length": max_len + 2,  # For start and stop symbols.
            "max_sequence_length_99ptile": max_len_99ptile + 2,  # For start and stop symbols.
            "pad_idx": pad_idx,
            "padding_symbol": padding_symbol,
            "unknown_symbol": unknown_symbol,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, backend):
        # TODO(1891): Remove backward compatibility hack once all models have been retrained with Ludwig after
        # https://github.com/ludwig-ai/ludwig/pull/1859.
        prefix = ""
        padding_symbol_metadata_key = "padding_symbol"
        unknown_symbol_metadata_key = "unknown_symbol"
        if "str2idx" not in metadata:
            prefix = "word_"
            padding_symbol_metadata_key = "word_pad_symbol"
            unknown_symbol_metadata_key = "word_unk_symbol"

        # ensure preprocessing param values match the metadata determined from dataset
        preprocessing_parameters["padding_symbol"] = metadata[padding_symbol_metadata_key]
        preprocessing_parameters["unknown_symbol"] = metadata[unknown_symbol_metadata_key]
        if preprocessing_parameters["fill_value"] == UNKNOWN_SYMBOL:
            preprocessing_parameters["fill_value"] = preprocessing_parameters["unknown_symbol"]
        if (
            "computed_fill_value" in preprocessing_parameters
            and preprocessing_parameters["computed_fill_value"] == UNKNOWN_SYMBOL
        ):
            preprocessing_parameters["computed_fill_value"] = preprocessing_parameters["unknown_symbol"]

        return build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata[f"{prefix}str2idx"],
            tokenizer_type=preprocessing_parameters[f"{prefix}tokenizer"],
            length_limit=metadata[f"{prefix}max_sequence_length"],
            padding_symbol=metadata[padding_symbol_metadata_key],
            padding=preprocessing_parameters["padding"],
            unknown_symbol=metadata[unknown_symbol_metadata_key],
            lowercase=preprocessing_parameters["lowercase"],
            tokenizer_vocab_file=preprocessing_parameters[f"{prefix}vocab_file"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )

    @staticmethod
    def add_feature_data(
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        proc_df[feature_config[PROC_COLUMN]] = TextFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]],
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


@register_input_feature(TEXT)
class TextInputFeature(TextFeatureMixin, SequenceInputFeature):
    encoder = {TYPE: "parallel_cnn", "max_sequence_length": None}

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature, encoder_obj=encoder_obj)
        self._input_shape = [feature[ENCODER]["max_sequence_length"]]

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert (
            inputs.dtype == torch.int8
            or inputs.dtype == torch.int16
            or inputs.dtype == torch.int32
            or inputs.dtype == torch.int64
        )
        assert len(inputs.shape) == 2

        inputs_mask = torch.not_equal(inputs, SpecialSymbol.PADDING.value)

        inputs_exp = inputs.type(torch.int32)
        lengths = torch.sum(inputs_mask.type(torch.int32), dim=1)
        encoder_output = self.encoder_obj(inputs_exp, mask=inputs_mask)
        encoder_output[LENGTHS] = lengths

        return encoder_output

    @property
    def input_dtype(self):
        return torch.int32

    @property
    def input_shape(self):
        return torch.Size(self._input_shape)

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        input_feature[ENCODER]["vocab"] = feature_metadata["idx2str"]
        input_feature[ENCODER]["max_sequence_length"] = feature_metadata["max_sequence_length"]
        input_feature[ENCODER]["pad_idx"] = feature_metadata["pad_idx"]
        input_feature[ENCODER]["num_tokens"] = len(feature_metadata["idx2str"])

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(input_feature, {TIED: None, ENCODER: {"type": "parallel_cnn"}})

        encoder_class = get_encoder_cls(input_feature["type"], input_feature[ENCODER]["type"])

        if hasattr(encoder_class, "default_params"):
            set_default_values(input_feature, encoder_class.default_params)

    @staticmethod
    def get_schema_cls():
        return TextInputFeatureConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _SequencePreprocessing(metadata)


@register_output_feature(TEXT)
class TextOutputFeature(TextFeatureMixin, SequenceOutputFeature):
    decoder = {TYPE: "generator", "max_sequence_length": 0, "vocab_size": 0}
    loss = {TYPE: "sequence_softmax_cross_entropy"}
    metric_functions = {LOSS: None, TOKEN_ACCURACY: None, LAST_ACCURACY: None, PERPLEXITY: None, EDIT_DISTANCE: None}
    default_validation_metric = LOSS

    def __init__(self, feature, output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)

    @classmethod
    def get_output_dtype(cls):
        return torch.int32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.decoder["max_sequence_length"]])

    @staticmethod
    def update_config_with_metadata(output_feature, feature_metadata, *args, **kwargs):
        output_feature[DECODER]["vocab_size"] = feature_metadata["vocab_size"]
        output_feature[DECODER]["max_sequence_length"] = feature_metadata["max_sequence_length"]
        if isinstance(output_feature[LOSS]["class_weights"], (list, tuple)):
            # [0, 0] for UNK and PAD
            output_feature[LOSS]["class_weights"] = [0, 0] + output_feature[LOSS]["class_weights"]
            if len(output_feature[LOSS]["class_weights"]) != output_feature[DECODER]["vocab_size"]:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({})".format(
                        len(output_feature[LOSS]["class_weights"]), output_feature[DECODER]["vocab_size"]
                    )
                )

        if output_feature[LOSS]["class_similarities_temperature"] > 0:
            if "class_similarities" in output_feature:
                distances = output_feature["class_similarities"]
                temperature = output_feature[LOSS]["class_similarities_temperature"]
                for i in range(len(distances)):
                    distances[i, :] = softmax(distances[i, :], temperature=temperature)
                output_feature[LOSS]["class_similarities"] = distances
            else:
                raise ValueError(
                    "class_similarities_temperature > 0,"
                    "but no class similarities are provided "
                    "for feature {}".format(output_feature[COLUMN])
                )

    @staticmethod
    def calculate_overall_stats(
        predictions,
        targets,
        train_set_metadata,
    ):
        return {}

    def postprocess_predictions(
        self,
        result,
        metadata,
    ):
        # todo: refactor to reuse SequenceOutputFeature.postprocess_predictions
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:

            def idx2str(pred):
                return [
                    metadata["idx2str"][token] if token < len(metadata["idx2str"]) else UNKNOWN_SYMBOL for token in pred
                ]

            result[predictions_col] = result[predictions_col].map(idx2str)

        last_preds_col = f"{self.feature_name}_{LAST_PREDICTIONS}"
        if last_preds_col in result:

            def last_idx2str(last_pred):
                if last_pred < len(metadata["idx2str"]):
                    return metadata["idx2str"][last_pred]
                return UNKNOWN_SYMBOL

            result[last_preds_col] = result[last_preds_col].map(last_idx2str)

        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        prob_col = f"{self.feature_name}_{PROBABILITY}"
        if probs_col in result:
            # currently does not return full probabilties because usually it is huge:
            # dataset x length x classes
            # TODO: add a mechanism for letting the user decide to save it
            result[probs_col] = result[probs_col].map(compute_token_probabilities)
            result[prob_col] = result[probs_col].map(
                partial(
                    compute_sequence_probability,
                    max_sequence_length=metadata["max_sequence_length"],
                    return_log_prob=True,
                ),
            )

        lengths_col = f"{self.feature_name}_{LENGTHS}"
        if lengths_col in result:
            del result[lengths_col]

        return result

    @staticmethod
    def create_postproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _SequencePostprocessing(metadata)

    @staticmethod
    def populate_defaults(output_feature):
        SequenceOutputFeature.populate_defaults(output_feature)

    @staticmethod
    def get_schema_cls():
        return TextOutputFeatureConfig

    def flatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(lambda x: x.flatten())
        return df

    def unflatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(
            lambda x: x.reshape(-1, self.decoder["max_sequence_length"]), meta=(probs_col, "object")
        )
        return df
