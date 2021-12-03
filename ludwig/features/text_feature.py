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

import numpy as np
import torch

from ludwig.constants import (
    COLUMN,
    EDIT_DISTANCE,
    FILL_WITH_CONST,
    LAST_ACCURACY,
    LAST_PREDICTIONS,
    LENGTHS,
    LOSS,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NAME,
    PERPLEXITY,
    PREDICTIONS,
    PROBABILITIES,
    PROBABILITY,
    PROC_COLUMN,
    SOFTMAX_CROSS_ENTROPY,
    TEXT,
    TIED,
    TOKEN_ACCURACY,
    TYPE,
)
from ludwig.encoders.registry import get_encoder_cls
from ludwig.features.sequence_feature import SequenceInputFeature, SequenceOutputFeature
from ludwig.utils.eval_utils import ConfusionMatrix
from ludwig.utils.math_utils import softmax
from ludwig.utils.misc_utils import set_default_value, set_default_values
from ludwig.utils.strings_utils import (
    build_sequence_matrix,
    create_vocabulary,
    PADDING_SYMBOL,
    tokenizer_registry,
    UNKNOWN_SYMBOL,
)
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)


class TextFeatureMixin:
    type = TEXT

    preprocessing_defaults = {
        "char_tokenizer": "characters",
        "char_vocab_file": None,
        "char_sequence_length_limit": 1024,
        "char_most_common": 70,
        "word_tokenizer": "space_punct",
        "pretrained_model_name_or_path": None,
        "word_vocab_file": None,
        "word_sequence_length_limit": 256,
        "word_most_common": 20000,
        "padding_symbol": PADDING_SYMBOL,
        "unknown_symbol": UNKNOWN_SYMBOL,
        "padding": "right",
        "lowercase": True,
        "missing_value_strategy": FILL_WITH_CONST,
        "fill_value": UNKNOWN_SYMBOL,
    }

    preprocessing_schema = {
        "char_tokenizer": {"type": "string", "enum": sorted(list(tokenizer_registry.keys()))},
        "char_vocab_file": {"type": ["string", "null"]},
        "char_sequence_length_limit": {"type": "integer", "minimum": 0},
        "char_most_common": {"type": "integer", "minimum": 0},
        "word_tokenizer": {"type": "string", "enum": sorted(list(tokenizer_registry.keys()))},
        "pretrained_model_name_or_path": {"type": ["string", "null"]},
        "word_vocab_file": {"type": ["string", "null"]},
        "word_sequence_length_limit": {"type": "integer", "minimum": 0},
        "word_most_common": {"type": "integer", "minimum": 0},
        "padding_symbol": {"type": "string"},
        "unknown_symbol": {"type": "string"},
        "padding": {"type": "string", "enum": ["right", "left"]},
        "lowercase": {"type": "boolean"},
        "missing_value_strategy": {"type": "string", "enum": MISSING_VALUE_STRATEGY_OPTIONS},
        "fill_value": {"type": "string"},
        "computed_fill_value": {"type": "string"},
    }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def feature_meta(column, preprocessing_parameters, backend):
        print(f"Called feature_meta: {column}")
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
        ) = create_vocabulary(
            column,
            tokenizer_type="characters",
            num_most_frequent=preprocessing_parameters["char_most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )
        (
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
        ) = create_vocabulary(
            column,
            tokenizer_type=preprocessing_parameters["word_tokenizer"],
            num_most_frequent=preprocessing_parameters["word_most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            vocab_file=preprocessing_parameters["word_vocab_file"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )
        return (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
        )

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        print(f"Called get_feature_meta: {column}")
        column = column.astype(str)
        tf_meta = TextFeatureMixin.feature_meta(column, preprocessing_parameters, backend)
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
        ) = tf_meta
        char_max_len = min(preprocessing_parameters["char_sequence_length_limit"], char_max_len)
        word_max_len = min(preprocessing_parameters["word_sequence_length_limit"], word_max_len)
        return {
            "char_idx2str": char_idx2str,
            "char_str2idx": char_str2idx,
            "char_str2freq": char_str2freq,
            "char_vocab_size": len(char_idx2str),
            "char_max_sequence_length": char_max_len,
            "char_pad_idx": char_pad_idx,
            "char_pad_symbol": char_pad_symbol,
            "char_unk_symbol": char_unk_symbol,
            "word_idx2str": word_idx2str,
            "word_str2idx": word_str2idx,
            "word_str2freq": word_str2freq,
            "word_vocab_size": len(word_idx2str),
            "word_max_sequence_length": word_max_len,
            "word_pad_idx": word_pad_idx,
            "word_pad_symbol": word_pad_symbol,
            "word_unk_symbol": word_unk_symbol,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, backend):
        print(f"Called feature_data: {column}")
        char_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata["char_str2idx"],
            tokenizer_type=preprocessing_parameters["char_tokenizer"],
            length_limit=metadata["char_max_sequence_length"],
            padding_symbol=metadata["char_pad_symbol"],
            padding=preprocessing_parameters["padding"],
            unknown_symbol=metadata["char_unk_symbol"],
            lowercase=preprocessing_parameters["lowercase"],
            tokenizer_vocab_file=preprocessing_parameters["char_vocab_file"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )
        word_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata["word_str2idx"],
            tokenizer_type=preprocessing_parameters["word_tokenizer"],
            length_limit=metadata["word_max_sequence_length"],
            padding_symbol=metadata["word_pad_symbol"],
            padding=preprocessing_parameters["padding"],
            unknown_symbol=metadata["word_unk_symbol"],
            lowercase=preprocessing_parameters["lowercase"],
            tokenizer_vocab_file=preprocessing_parameters["word_vocab_file"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            processor=backend.df_engine,
        )

        return char_data, word_data

    @staticmethod
    def add_feature_data(
        feature, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        chars_data, words_data = TextFeatureMixin.feature_data(
            input_df[feature[COLUMN]].astype(str), metadata[feature[NAME]], preprocessing_parameters, backend
        )
        proc_df[f"{feature[PROC_COLUMN]}_char"] = chars_data
        proc_df[f"{feature[PROC_COLUMN]}_word"] = words_data
        return proc_df


class TextInputFeature(TextFeatureMixin, SequenceInputFeature):
    encoder = "parallel_cnn"
    max_sequence_length = None
    level = "word"

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature, encoder_obj=encoder_obj)
        if "pad_idx" in feature.keys():
            self.pad_idx = feature["pad_idx"]
        else:
            self.pad_idx = None
        self._input_shape = [feature["max_sequence_length"]]

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert (
            inputs.dtype == torch.int8
            or inputs.dtype == torch.int16
            or inputs.dtype == torch.int32
            or inputs.dtype == torch.int64
        )
        assert len(inputs.shape) == 2

        if self.pad_idx is not None:
            inputs_mask = torch.not_equal(inputs, self.pad_idx)
        else:
            inputs_mask = torch.not_equal(inputs, 0)

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
        input_feature["vocab"] = feature_metadata[input_feature["level"] + "_idx2str"]
        input_feature["max_sequence_length"] = feature_metadata[input_feature["level"] + "_max_sequence_length"]
        input_feature["pad_idx"] = feature_metadata[input_feature["level"] + "_pad_idx"]
        input_feature["num_tokens"] = len(feature_metadata[input_feature["level"] + "_idx2str"])

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(input_feature, {TIED: None, "encoder": "parallel_cnn", "level": "word"})

        encoder_class = get_encoder_cls(input_feature["type"], input_feature["encoder"])

        if hasattr(encoder_class, "default_params"):
            set_default_values(input_feature, encoder_class.default_params)

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape


class TextOutputFeature(TextFeatureMixin, SequenceOutputFeature):
    loss = {TYPE: SOFTMAX_CROSS_ENTROPY}
    metric_functions = {LOSS: None, TOKEN_ACCURACY: None, LAST_ACCURACY: None, PERPLEXITY: None, EDIT_DISTANCE: None}
    default_validation_metric = LOSS
    max_sequence_length = 0
    num_classes = 0
    level = "word"

    def __init__(self, feature):
        super().__init__(feature)

    @classmethod
    def get_output_dtype(cls):
        return torch.int32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    def overall_statistics_metadata(self):
        return {"level": self.level}

    @staticmethod
    def update_config_with_metadata(output_feature, feature_metadata, *args, **kwargs):
        output_feature["num_classes"] = feature_metadata["{}_vocab_size".format(output_feature["level"])]
        output_feature["max_sequence_length"] = feature_metadata[
            "{}_max_sequence_length".format(output_feature["level"])
        ]
        if isinstance(output_feature[LOSS]["class_weights"], (list, tuple)):
            # [0, 0] for UNK and PAD
            output_feature[LOSS]["class_weights"] = [0, 0] + output_feature[LOSS]["class_weights"]
            if len(output_feature[LOSS]["class_weights"]) != output_feature["num_classes"]:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({})".format(
                        len(output_feature[LOSS]["class_weights"]), output_feature["num_classes"]
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

        if output_feature[LOSS][TYPE] == "sampled_softmax_cross_entropy":
            level_str2freq = "{}_str2freq".format(output_feature["level"])
            level_idx2str = "{}_idx2str".format(output_feature["level"])
            output_feature[LOSS]["class_counts"] = [
                feature_metadata[level_str2freq][cls] for cls in feature_metadata[level_idx2str]
            ]

    @staticmethod
    def calculate_overall_stats(
        predictions,
        targets,
        train_set_metadata,
    ):
        overall_stats = {}
        level_idx2str = "{}_{}".format(train_set_metadata["level"], "idx2str")

        sequences = targets
        last_elem_sequence = sequences[np.arange(sequences.shape[0]), (sequences != 0).cumsum(1).argmax(1)]
        confusion_matrix = ConfusionMatrix(
            last_elem_sequence, predictions[LAST_PREDICTIONS], labels=train_set_metadata[level_idx2str]
        )
        overall_stats["confusion_matrix"] = confusion_matrix.cm.tolist()
        overall_stats["overall_stats"] = confusion_matrix.stats()
        overall_stats["per_class_stats"] = confusion_matrix.per_class_stats()

        return overall_stats

    def postprocess_predictions(
        self,
        result,
        metadata,
        output_directory,
        backend,
    ):
        # todo: refactor to reuse SequenceOutputFeature.postprocess_predictions
        level_idx2str = f"{self.level}_idx2str"

        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            if level_idx2str in metadata:

                def idx2str(pred):
                    return [
                        metadata[level_idx2str][token] if token < len(metadata[level_idx2str]) else UNKNOWN_SYMBOL
                        for token in pred
                    ]

                result[predictions_col] = backend.df_engine.map_objects(result[predictions_col], idx2str)

        last_preds_col = f"{self.feature_name}_{LAST_PREDICTIONS}"
        if last_preds_col in result:
            if level_idx2str in metadata:

                def last_idx2str(last_pred):
                    if last_pred < len(metadata[level_idx2str]):
                        return metadata[level_idx2str][last_pred]
                    return UNKNOWN_SYMBOL

                result[last_preds_col] = backend.df_engine.map_objects(result[last_preds_col], last_idx2str)

        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        prob_col = f"{self.feature_name}_{PROBABILITY}"
        if probs_col in result:

            def compute_prob(probs):
                if isinstance(probs, (list, tuple, np.ndarray)):
                    for i in range(len(probs)):
                        probs[i] = np.max(probs[i])
                    return np.prod(probs)
                else:
                    return np.prod(probs, axis=-1)

            result[prob_col] = backend.df_engine.map_objects(
                result[probs_col],
                compute_prob,
            )

            # commenting probabilities out because usually it is huge:
            # dataset x length x classes
            # todo: add a mechanism for letting the user decide to save it
            # result[probs_col] = backend.df_engine.map_objects(
            #     result[probs_col],
            #     lambda prob: np.amax(prob, axis=-1),
            # )

        lengths_col = f"{self.feature_name}_{LENGTHS}"
        if lengths_col in result:
            del result[lengths_col]

        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, "level", "word")
        SequenceOutputFeature.populate_defaults(output_feature)

    def flatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(lambda x: x.flatten())
        return df

    def unflatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(
            lambda x: x.reshape(-1, self.max_sequence_length), meta=(probs_col, "object")
        )
        return df
