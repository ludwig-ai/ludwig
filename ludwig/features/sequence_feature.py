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
    SEQUENCE,
    SEQUENCE_ACCURACY,
    SUM,
    TIED,
    TOKEN_ACCURACY,
    TYPE,
)
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.utils import output_feature_utils
from ludwig.utils.math_utils import softmax
from ludwig.utils.misc_utils import get_from_registry, set_default_value
from ludwig.utils.strings_utils import (
    build_sequence_matrix,
    create_vocabulary,
    PADDING_SYMBOL,
    SpecialSymbol,
    START_SYMBOL,
    STOP_SYMBOL,
    tokenizer_registry,
    UNKNOWN_SYMBOL,
)
from ludwig.utils.tokenizers import TORCHSCRIPT_COMPATIBLE_TOKENIZERS
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)


class _SequencePreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by SequenceFeatureMixin.add_feature_data."""

    def __init__(self, metadata: Dict[str, Any]):
        super().__init__()
        if metadata["preprocessing"]["tokenizer"] not in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
            raise ValueError(
                f"{metadata['preprocessing']['tokenizer']} is not supported by torchscript. Please use "
                f"one of {TORCHSCRIPT_COMPATIBLE_TOKENIZERS}."
            )

        self.lowercase = metadata["preprocessing"]["lowercase"]
        self.tokenizer = get_from_registry(metadata["preprocessing"]["tokenizer"], tokenizer_registry)(
            pretrained_model_name_or_path=metadata["preprocessing"].get("pretrained_model_name_or_path", None)
        )
        self.padding_symbol = metadata["preprocessing"]["padding_symbol"]
        self.unknown_symbol = metadata["preprocessing"]["unknown_symbol"]
        self.start_symbol = START_SYMBOL
        self.stop_symbol = STOP_SYMBOL
        self.max_sequence_length = int(metadata["max_sequence_length"])
        self.unit_to_id = metadata["str2idx"]

    def forward(self, v: Union[List[str], List[torch.Tensor], torch.Tensor]):
        """Takes a list of strings and returns a tensor of token ids."""
        if not torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        if self.lowercase:
            sequences = [sequence.lower() for sequence in v]
        else:
            sequences = v

        unit_sequences = self.tokenizer(sequences)
        # refines type of unit_sequences from Any to List[List[str]]
        assert torch.jit.isinstance(unit_sequences, List[List[str]]), "unit_sequences is not a list of lists."

        sequence_matrix = torch.full(
            [len(unit_sequences), self.max_sequence_length], self.unit_to_id[self.padding_symbol]
        )
        sequence_matrix[:, 0] = self.unit_to_id[self.start_symbol]
        for sample_idx, unit_sequence in enumerate(unit_sequences):
            # Add <EOS> if sequence length is less than max_sequence_length. Else, truncate to max_sequence_length.
            if len(unit_sequence) + 1 < self.max_sequence_length:
                sequence_length = len(unit_sequence)
                sequence_matrix[sample_idx][len(unit_sequence) + 1] = self.unit_to_id[self.stop_symbol]
            else:
                sequence_length = self.max_sequence_length - 1

            for i in range(sequence_length):
                curr_unit = unit_sequence[i]
                if curr_unit in self.unit_to_id:
                    curr_id = self.unit_to_id[curr_unit]
                else:
                    curr_id = self.unit_to_id[self.unknown_symbol]
                sequence_matrix[sample_idx][i + 1] = curr_id

        return sequence_matrix


class _SequencePredict(PredictModule):
    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)
        probabilities = torch.softmax(logits, -1)
        predictions = torch.argmax(logits, -1)

        # predictions: [batch_size, sequence_length]
        # probabilities: [batch_size, sequence_length, vocab_size]
        # logits: [batch_size, sequence_length, vocab_size]
        return {self.predictions_key: predictions, self.probabilities_key: probabilities, self.logits_key: logits}


class SequenceFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return SEQUENCE

    @staticmethod
    def preprocessing_defaults():
        return {
            "max_sequence_length": 256,
            "most_common": 20000,
            "padding_symbol": PADDING_SYMBOL,
            "unknown_symbol": UNKNOWN_SYMBOL,
            "padding": "right",
            "tokenizer": "space",
            "lowercase": False,
            "vocab_file": None,
            "missing_value_strategy": FILL_WITH_CONST,
            "fill_value": UNKNOWN_SYMBOL,
        }

    @staticmethod
    def preprocessing_schema():
        return {
            "max_sequence_length": {"type": "integer", "minimum": 0},
            "most_common": {"type": "integer", "minimum": 0},
            "padding_symbol": {"type": "string"},
            "unknown_symbol": {"type": "string"},
            "padding": {"type": "string", "enum": ["right", "left"]},
            "tokenizer": {"type": "string", "enum": sorted(list(tokenizer_registry.keys()))},
            "lowercase": {"type": "boolean"},
            "vocab_file": {"type": ["string", "null"]},
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
        idx2str, str2idx, str2freq, max_length, _, _, _, _ = create_vocabulary(
            column,
            preprocessing_parameters["tokenizer"],
            lowercase=preprocessing_parameters["lowercase"],
            num_most_frequent=preprocessing_parameters["most_common"],
            vocab_file=preprocessing_parameters["vocab_file"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            processor=backend.df_engine,
        )
        max_length = min(preprocessing_parameters["max_sequence_length"], max_length)
        return {
            "idx2str": idx2str,
            "str2idx": str2idx,
            "str2freq": str2freq,
            "vocab_size": len(idx2str),
            "max_sequence_length": max_length + 2,  # For start and end symbol.
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, backend):
        sequence_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata["str2idx"],
            tokenizer_type=preprocessing_parameters["tokenizer"],
            length_limit=metadata["max_sequence_length"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            padding=preprocessing_parameters["padding"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            lowercase=preprocessing_parameters["lowercase"],
            tokenizer_vocab_file=preprocessing_parameters["vocab_file"],
            processor=backend.df_engine,
        )
        return sequence_data

    @staticmethod
    def add_feature_data(
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        sequence_data = SequenceInputFeature.feature_data(
            input_df[feature_config[COLUMN]].astype(str),
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        proc_df[feature_config[PROC_COLUMN]] = sequence_data
        return proc_df


class SequenceInputFeature(SequenceFeatureMixin, InputFeature):
    encoder = "embed"
    max_sequence_length = None

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        # TODO: Potentially abstract this feature-specific attribute overwrite to a consolidated design.
        if "vocab" in feature:
            feature["vocab_size"] = len(feature["vocab"])
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs: torch.Tensor, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.int8, inputs.dtype, torch.int16, torch.int32, torch.int64]
        assert len(inputs.shape) == 2
        inputs_exp = inputs.type(torch.int32)
        inputs_mask = torch.not_equal(inputs, SpecialSymbol.PADDING.value)
        lengths = torch.sum(inputs_mask.type(torch.int32), dim=1)
        encoder_output = self.encoder_obj(inputs_exp, mask=inputs_mask)
        encoder_output[LENGTHS] = lengths
        return encoder_output

    @property
    def input_dtype(self):
        return torch.int32

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        input_feature["vocab"] = feature_metadata["idx2str"]
        input_feature["max_sequence_length"] = feature_metadata["max_sequence_length"]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, "encoder", "parallel_cnn")

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _SequencePreprocessing(metadata)


class SequenceOutputFeature(SequenceFeatureMixin, OutputFeature):
    decoder = "generator"
    loss = {TYPE: "sequence_softmax_cross_entropy"}
    metric_functions = {
        LOSS: None,
        TOKEN_ACCURACY: None,
        SEQUENCE_ACCURACY: None,
        LAST_ACCURACY: None,
        PERPLEXITY: None,
        EDIT_DISTANCE: None,
    }
    default_validation_metric = LOSS
    max_sequence_length = 0
    num_classes = 0

    def __init__(self, feature: Dict[str, Any], output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs: Dict[str, torch.Tensor], target=None):
        return self.decoder_obj(inputs, target=target)

    def create_predict_module(self) -> PredictModule:
        return _SequencePredict()

    def get_prediction_set(self):
        return self.decoder_obj.get_prediction_set()

    @classmethod
    def get_output_dtype(cls):
        return torch.int32

    @property
    def input_shape(self) -> torch.Size:
        # Dummy implementation.
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @staticmethod
    def update_config_with_metadata(output_feature, feature_metadata, *args, **kwargs):
        output_feature["vocab_size"] = feature_metadata["vocab_size"]
        output_feature["max_sequence_length"] = feature_metadata["max_sequence_length"]
        if isinstance(output_feature[LOSS]["class_weights"], (list, tuple)):
            if len(output_feature[LOSS]["class_weights"]) != output_feature["vocab_size"]:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({}) for feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and their order and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        len(output_feature[LOSS]["class_weights"]),
                        output_feature["vocab_size"],
                        output_feature[COLUMN],
                    )
                )

        if output_feature[LOSS]["class_similarities_temperature"] > 0:
            if "class_similarities" in output_feature[LOSS]:
                similarities = output_feature[LOSS]["class_similarities"]
                temperature = output_feature[LOSS]["class_similarities_temperature"]

                curr_row = 0
                first_row_length = 0
                is_first_row = True
                for row in similarities:
                    if is_first_row:
                        first_row_length = len(row)
                        is_first_row = False
                        curr_row += 1
                    else:
                        curr_row_length = len(row)
                        if curr_row_length != first_row_length:
                            raise ValueError(
                                "The length of row {} of the class_similarities "
                                "of {} is {}, different from the length of "
                                "the first row {}. All rows must have "
                                "the same length.".format(
                                    curr_row, output_feature[COLUMN], curr_row_length, first_row_length
                                )
                            )
                        else:
                            curr_row += 1
                all_rows_length = first_row_length

                if all_rows_length != len(similarities):
                    raise ValueError(
                        "The class_similarities matrix of {} has "
                        "{} rows and {} columns, "
                        "their number must be identical.".format(
                            output_feature[COLUMN], len(similarities), all_rows_length
                        )
                    )

                if all_rows_length != output_feature["vocab_size"]:
                    raise ValueError(
                        "The size of the class_similarities matrix of {} is "
                        "{}, different from the number of classes ({}). "
                        "Check the metadata JSON file to see the classes "
                        "and their order and "
                        "consider <UNK> and <PAD> class too.".format(
                            output_feature[COLUMN], all_rows_length, output_feature["vocab_size"]
                        )
                    )

                similarities = np.array(similarities, dtype=np.float32)
                for i in range(len(similarities)):
                    similarities[i, :] = softmax(similarities[i, :], temperature=temperature)
                output_feature[LOSS]["class_similarities"] = similarities
            else:
                raise ValueError(
                    "class_similarities_temperature > 0, "
                    "but no class_similarities are provided "
                    "for feature {}".format(output_feature[COLUMN])
                )

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        # TODO(Justin): Add a confusion matrix, see
        # https://github.com/ludwig-ai/ludwig/blob/tf-legacy/ludwig/features/sequence_feature.py#L411
        return {}

    def postprocess_predictions(
        self,
        result,
        metadata,
        output_directory,
        backend,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        lengths_col = f"{self.feature_name}_{LENGTHS}"
        if predictions_col in result:
            if "idx2str" in metadata:

                def idx2str(row):
                    pred = row[predictions_col]
                    length = metadata["max_sequence_length"]
                    return [
                        metadata["idx2str"][token] if token < len(metadata["idx2str"]) else UNKNOWN_SYMBOL
                        for token in [pred[i] for i in range(length)]
                    ]

                result[predictions_col] = backend.df_engine.apply_objects(result, idx2str)

        last_preds_col = f"{self.feature_name}_{LAST_PREDICTIONS}"
        if last_preds_col in result:
            if "idx2str" in metadata:

                def last_idx2str(last_pred):
                    if last_pred < len(metadata["idx2str"]):
                        return metadata["idx2str"][last_pred]
                    return UNKNOWN_SYMBOL

                result[last_preds_col] = backend.df_engine.map_objects(result[last_preds_col], last_idx2str)

        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        if probs_col in result:

            def token_prob(prob):
                dim = len(prob.shape)
                if dim != 2:
                    # probs should be shape [s, nc]
                    raise ValueError(
                        f"Sequence probability array should be 2-dimensional "
                        f"shape, instead shape is {dim}-dimensional ({prob.shape})"
                    )
                return np.amax(prob, axis=-1)

            # get probability of token in that sequence position
            result[probs_col] = backend.df_engine.map_objects(result[probs_col], token_prob)

            def compute_log_prob(row):
                # sum log probability for tokens up to sequence length
                # create mask only tokens for sequence length
                seq_prob = row[probs_col]
                length = metadata["max_sequence_length"]
                mask = np.arange(seq_prob.shape[-1]) < np.array(length).reshape(-1, 1)
                return np.sum(np.log(seq_prob) * mask, axis=-1)[0]

            # commenting probabilities out because usually it is huge:
            # dataset x length x classes
            # todo: add a mechanism for letting the user decide to save it
            probability_col = f"{self.feature_name}_{PROBABILITY}"
            result[probability_col] = backend.df_engine.apply_objects(result, compute_log_prob)

        if lengths_col in result:
            del result[lengths_col]

        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {
                TYPE: "sequence_softmax_cross_entropy",
                "class_weights": 1,
                "robust_lambda": 0,
                "confidence_penalty": 0,
                "class_similarities_temperature": 0,
                "weight": 1,
            },
        )

        set_default_value(output_feature[LOSS], "unique", False)
        set_default_value(output_feature, "decoder", "generator")

        if output_feature["decoder"] == "tagger":
            set_default_value(output_feature, "reduce_input", None)

        set_default_value(output_feature, "dependencies", [])
        set_default_value(output_feature, "reduce_input", SUM)
        set_default_value(output_feature, "reduce_dependencies", SUM)

    def flatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(lambda x: x.flatten())
        return df

    def unflatten(self, df: DataFrame) -> DataFrame:
        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        df[probs_col] = df[probs_col].apply(lambda x: x.reshape(-1, self.num_classes), meta=(probs_col, "object"))
        return df
