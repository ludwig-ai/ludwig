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
from typing import Dict, List, Union

import numpy as np
import torch

from ludwig.constants import (
    COLUMN,
    LAST_PREDICTIONS,
    LENGTHS,
    NAME,
    PREDICTIONS,
    PROBABILITIES,
    PROBABILITY,
    PROC_COLUMN,
    SEQUENCE,
)
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.features.feature_utils import compute_sequence_probability, compute_token_probabilities
from ludwig.schema.features.sequence_feature import SequenceInputFeatureConfig, SequenceOutputFeatureConfig
from ludwig.types import (
    FeatureMetadataDict,
    FeaturePostProcessingOutputDict,
    PreprocessingConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils import output_feature_utils
from ludwig.utils.math_utils import softmax
from ludwig.utils.strings_utils import (
    build_sequence_matrix,
    create_vocabulary,
    SpecialSymbol,
    START_SYMBOL,
    STOP_SYMBOL,
    UNKNOWN_SYMBOL,
)
from ludwig.utils.tokenizers import get_tokenizer_from_registry
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class _SequencePreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by SequenceFeatureMixin.add_feature_data."""

    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.lowercase = metadata["preprocessing"]["lowercase"]
        self.tokenizer_type = metadata["preprocessing"]["tokenizer"]
        self.tokenizer = get_tokenizer_from_registry(self.tokenizer_type)(
            pretrained_model_name_or_path=metadata["preprocessing"].get("pretrained_model_name_or_path", None)
        )

        if not isinstance(self.tokenizer, torch.nn.Module):
            raise ValueError(f"tokenizer must be a torch.nn.Module, got {self.tokenizer}")

        self.padding_symbol = metadata["preprocessing"]["padding_symbol"]
        self.unknown_symbol = metadata["preprocessing"]["unknown_symbol"]
        self.start_symbol = START_SYMBOL
        self.stop_symbol = STOP_SYMBOL
        self.max_sequence_length = int(metadata["max_sequence_length"])
        self.unit_to_id = metadata["str2idx"]
        self.computed_fill_value = metadata["preprocessing"]["computed_fill_value"]

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        """Takes a list of strings and returns a tensor of token ids."""
        if not torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        futures: List[torch.jit.Future[torch.Tensor]] = []
        for sequence in v:
            futures.append(
                torch.jit.fork(
                    self._process_sequence,
                    sequence,
                )
            )

        sequence_matrix = []
        for future in futures:
            sequence_matrix.append(torch.jit.wait(future))

        return torch.stack(sequence_matrix)

    def _process_sequence(self, sequence: str) -> torch.Tensor:
        sequence = self.computed_fill_value if sequence == "nan" else sequence

        # If tokenizer is HF, we defer lowercase transformation to the tokenizer.
        if self.lowercase and self.tokenizer_type != "hf_tokenizer":
            sequence_str: str = sequence.lower()
        else:
            sequence_str: str = sequence

        sequence_vector = torch.full([self.max_sequence_length], self.unit_to_id[self.padding_symbol])

        if self.tokenizer_type == "hf_tokenizer":
            # Handles start, stop, and unknown symbols implicitly
            unit_sequence = self.tokenizer(sequence)
            assert torch.jit.isinstance(unit_sequence, List[int])
            # Ensures that the sequence lengths are aligned between the input and output tensors.
            sequence_length = min(len(unit_sequence), self.max_sequence_length)
            sequence_vector[:sequence_length] = torch.tensor(unit_sequence)[:sequence_length]
            return sequence_vector

        # If tokenizer is not HF, we manually convert tokens to IDs and insert start, stop, and unknown symbols.
        unit_sequence = self.tokenizer(sequence_str)
        assert torch.jit.isinstance(unit_sequence, List[str])

        sequence_vector[0] = self.unit_to_id[self.start_symbol]
        if len(unit_sequence) + 1 < self.max_sequence_length:
            sequence_length = len(unit_sequence)
            sequence_vector[len(unit_sequence) + 1] = self.unit_to_id[self.stop_symbol]
        else:
            sequence_length = self.max_sequence_length - 1

        for i in range(sequence_length):
            curr_unit = unit_sequence[i]
            if curr_unit in self.unit_to_id:
                curr_id = self.unit_to_id[curr_unit]
            else:
                curr_id = self.unit_to_id[self.unknown_symbol]
            sequence_vector[i + 1] = curr_id
        return sequence_vector


class _SequencePostprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.max_sequence_length = int(metadata["max_sequence_length"])
        self.idx2str = metadata["idx2str"]
        self.unknown_symbol = UNKNOWN_SYMBOL
        self.predictions_key = PREDICTIONS
        self.probabilities_key = PROBABILITIES
        self.probability_key = PROBABILITY

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> FeaturePostProcessingOutputDict:
        pred_predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)
        pred_probabilities = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.probabilities_key)

        predictions: List[List[str]] = []
        for sequence in pred_predictions:
            sequence_predictions: List[str] = []
            for i in range(self.max_sequence_length):
                unit_id = int(sequence[i].item())
                if unit_id < len(self.idx2str):
                    unit_prediction = self.idx2str[unit_id]
                else:
                    unit_prediction = self.unknown_symbol
                sequence_predictions.append(unit_prediction)
            predictions.append(sequence_predictions)

        probabilities, _ = torch.max(pred_probabilities, dim=-1)
        probability = torch.sum(torch.log(probabilities), dim=-1)

        return {
            self.predictions_key: predictions,
            self.probabilities_key: probabilities,
            self.probability_key: probability,
        }


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
    def cast_column(column, backend):
        return column.astype(str)

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        vocabulary = create_vocabulary(
            column,
            preprocessing_parameters["tokenizer"],
            lowercase=preprocessing_parameters["lowercase"],
            num_most_frequent=preprocessing_parameters["most_common"],
            vocab_file=preprocessing_parameters["vocab_file"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            ngram_size=preprocessing_parameters["ngram_size"],
            processor=backend.df_engine,
        )
        logger.info(
            f"Max length of feature '{column.name}': {vocabulary.line_length_max} (without start and stop symbols)"
        )

        # Use sequence_length if provided, otherwise use max length found in dataset.
        if preprocessing_parameters["sequence_length"] is not None:
            logger.info(
                f"Setting max length to sequence_length={preprocessing_parameters['sequence_length']} provided in "
                f"preprocessing parameters"
            )
            max_sequence_length = preprocessing_parameters["sequence_length"]
        else:
            max_sequence_length = vocabulary.line_length_max + 2  # For start and stop symbols.
            logger.info(f"Setting max length using dataset: {max_sequence_length} (including start and stop symbols)")

            # If max_sequence_length is None, then use the max length found in the dataset.
            if (
                preprocessing_parameters["max_sequence_length"] is not None
                and preprocessing_parameters["max_sequence_length"] < max_sequence_length
            ):
                logger.info(
                    f"Truncating max length with max_sequence_length={preprocessing_parameters['max_sequence_length']} "
                    f"from preprocessing parameters"
                )
                max_sequence_length = preprocessing_parameters["max_sequence_length"]

        logger.info(f"max sequence length is {max_sequence_length} for feature '{column.name}'")
        return {
            "idx2str": vocabulary.vocab,
            "str2idx": vocabulary.str2idx,
            "str2freq": vocabulary.str2freq,
            "vocab_size": len(vocabulary.vocab),
            "max_sequence_length": max_sequence_length,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters: PreprocessingConfigDict, backend):
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
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        sequence_data = SequenceInputFeature.feature_data(
            input_df[feature_config[COLUMN]],
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        proc_df[feature_config[PROC_COLUMN]] = sequence_data
        return proc_df


class SequenceInputFeature(SequenceFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: SequenceInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

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
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.vocab = feature_metadata["idx2str"]
        feature_config.encoder.vocab_size = len(feature_metadata["idx2str"])
        feature_config.encoder.max_sequence_length = feature_metadata["max_sequence_length"]

    @staticmethod
    def get_schema_cls():
        return SequenceInputFeatureConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.encoder_obj.config.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SequencePreprocessing(metadata)


class SequenceOutputFeature(SequenceFeatureMixin, OutputFeature):
    def __init__(
        self,
        output_feature_config: Union[SequenceOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        super().__init__(output_feature_config, output_features, **kwargs)
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
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
        return torch.Size([self.decoder_obj.config.max_sequence_length])

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.decoder.vocab_size = feature_metadata["vocab_size"]
        feature_config.decoder.max_sequence_length = feature_metadata["max_sequence_length"]
        if isinstance(feature_config.loss.class_weights, (list, tuple)):
            if len(feature_config.loss.class_weights) != feature_config.decoder.vocab_size:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({}) for feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and their order and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        len(feature_config.loss.class_weights),
                        feature_config.decoder.vocab_size,
                        feature_config.column,
                    )
                )

        if isinstance(feature_config.loss.class_weights, dict):
            if feature_metadata["str2idx"].keys() != feature_config.loss.class_weights.keys():
                raise ValueError(
                    "The class_weights keys ({}) are not compatible with "
                    "the classes ({}) of feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and consider there needs to be a weight "
                    "for the <UNK> class too.".format(
                        feature_config.loss.class_weights.keys(),
                        feature_metadata["str2idx"].keys(),
                        feature_config.column,
                    )
                )
            else:
                class_weights = feature_config.loss.class_weights
                idx2str = feature_metadata["idx2str"]
                class_weights_list = [class_weights[s] for s in idx2str]
                feature_config.loss.class_weights = class_weights_list

        if feature_config.loss.class_similarities_temperature > 0:
            if "class_similarities" in feature_config.loss:
                similarities = feature_config.loss.class_similarities
                temperature = feature_config.loss.class_similarities_temperature

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
                                    curr_row, feature_config.column, curr_row_length, first_row_length
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
                            feature_config.column, len(similarities), all_rows_length
                        )
                    )

                if all_rows_length != feature_config.decoder.vocab_size:
                    raise ValueError(
                        "The size of the class_similarities matrix of {} is "
                        "{}, different from the number of classes ({}). "
                        "Check the metadata JSON file to see the classes "
                        "and their order and "
                        "consider <UNK> and <PAD> class too.".format(
                            feature_config.column, all_rows_length, feature_config.decoder.vocab_size
                        )
                    )

                similarities = np.array(similarities, dtype=np.float32)
                for i in range(len(similarities)):
                    similarities[i, :] = softmax(similarities[i, :], temperature=temperature)
                feature_config.loss.class_similarities = similarities
            else:
                raise ValueError(
                    "class_similarities_temperature > 0, "
                    "but no class_similarities are provided "
                    "for feature {}".format(feature_config.column)
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

                result[predictions_col] = result.apply(idx2str, axis=1)

        last_preds_col = f"{self.feature_name}_{LAST_PREDICTIONS}"
        if last_preds_col in result:
            if "idx2str" in metadata:

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
                )
            )

        if lengths_col in result:
            del result[lengths_col]

        return result

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SequencePostprocessing(metadata)

    @staticmethod
    def get_schema_cls():
        return SequenceOutputFeatureConfig
