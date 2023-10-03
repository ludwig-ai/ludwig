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
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from ludwig.constants import (
    COLUMN,
    IGNORE_INDEX_TOKEN_ID,
    LAST_PREDICTIONS,
    LENGTHS,
    NAME,
    PREDICTIONS,
    PREPROCESSING,
    PROBABILITIES,
    PROBABILITY,
    PROC_COLUMN,
    RESPONSE,
    TEXT,
)
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature
from ludwig.features.feature_utils import compute_sequence_probability, compute_token_probabilities
from ludwig.features.sequence_feature import (
    _SequencePostprocessing,
    _SequencePreprocessing,
    SequenceInputFeature,
    SequenceOutputFeature,
)
from ludwig.modules.metric_registry import get_metric_tensor_input
from ludwig.schema.features.text_feature import TextInputFeatureConfig, TextOutputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.math_utils import softmax
from ludwig.utils.strings_utils import (
    build_sequence_matrix,
    create_vocabulary,
    get_tokenizer,
    SpecialSymbol,
    UNKNOWN_SYMBOL,
    Vocabulary,
)

logger = logging.getLogger(__name__)


def get_decoded_targets_and_predictions(
    targets: Tensor,
    predictions: Dict[str, Tensor],
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[str], List[str]]:
    """Returns the decoded targets and predictions, accounting for IGNORE_INDEX_TOKEN_ID."""
    sanitized_targets = torch.where(targets != IGNORE_INDEX_TOKEN_ID, targets, tokenizer.pad_token_id)
    sanitized_predictions = torch.where(
        predictions[PREDICTIONS] != IGNORE_INDEX_TOKEN_ID,
        predictions[PREDICTIONS],
        tokenizer.pad_token_id,
    )
    decoded_targets = tokenizer.batch_decode(sanitized_targets, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(sanitized_predictions, skip_special_tokens=True)
    return decoded_targets, decoded_predictions


class TextFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return TEXT

    @staticmethod
    def cast_column(column, backend):
        return column.astype(str)

    @staticmethod
    def feature_meta(column, preprocessing_parameters: PreprocessingConfigDict, backend) -> Vocabulary:
        return create_vocabulary(
            column,
            tokenizer_type=preprocessing_parameters["tokenizer"],
            num_most_frequent=preprocessing_parameters["most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            vocab_file=preprocessing_parameters["vocab_file"],
            unknown_symbol=preprocessing_parameters["unknown_symbol"],
            padding_symbol=preprocessing_parameters["padding_symbol"],
            pretrained_model_name_or_path=preprocessing_parameters["pretrained_model_name_or_path"],
            ngram_size=preprocessing_parameters["ngram_size"],
            compute_idf=preprocessing_parameters["compute_idf"],
            processor=backend.df_engine,
        )

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        vocabulary = TextFeatureMixin.feature_meta(column, preprocessing_parameters, backend)
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
            max_sequence_length_99ptile = preprocessing_parameters["sequence_length"]
        else:
            max_sequence_length = vocabulary.line_length_max + 2  # For start and stop symbols.
            max_sequence_length_99ptile = vocabulary.line_length_99ptile + 2  # For start and stop symbols.
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
                max_sequence_length_99ptile = min(vocabulary.line_length_99ptile, max_sequence_length)

        logger.info(f"max sequence length is {max_sequence_length} for feature '{column.name}'")

        return {
            "idx2str": vocabulary.vocab,
            "str2idx": vocabulary.str2idx,
            "str2freq": vocabulary.str2freq,
            "str2idf": vocabulary.str2idf,
            "vocab_size": len(vocabulary.vocab),
            "max_sequence_length": max_sequence_length,
            "max_sequence_length_99ptile": max_sequence_length_99ptile,
            "pad_idx": vocabulary.pad_idx,
            "padding_symbol": vocabulary.padding_symbol,
            "unknown_symbol": vocabulary.unknown_symbol,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters: PreprocessingConfigDict, backend):
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

        sequences = column

        return build_sequence_matrix(
            sequences=sequences,
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
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        proc_df[feature_config[PROC_COLUMN]] = TextFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]],
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


class TextInputFeature(TextFeatureMixin, SequenceInputFeature):
    def __init__(self, input_feature_config: TextInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, encoder_obj=encoder_obj, **kwargs)

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
        return torch.Size([self.encoder_obj.config.max_sequence_length])

    def update_config_after_module_init(self, feature_config):
        feature_config.encoder = self.encoder_obj.config

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.vocab = feature_metadata["idx2str"]
        feature_config.encoder.vocab_size = len(feature_metadata["idx2str"])
        feature_config.encoder.max_sequence_length = feature_metadata["max_sequence_length"]
        feature_config.encoder.pad_idx = feature_metadata["pad_idx"]
        feature_config.encoder.num_tokens = len(feature_metadata["idx2str"])
        feature_config.encoder.str2freq = feature_metadata["str2freq"]
        feature_config.encoder.str2idf = feature_metadata["str2idf"]
        feature_config.encoder.skip = feature_metadata[PREPROCESSING].get("cache_encoder_embeddings", False)

    @staticmethod
    def get_schema_cls():
        return TextInputFeatureConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SequencePreprocessing(metadata)


class TextOutputFeature(TextFeatureMixin, SequenceOutputFeature):
    def __init__(
        self,
        output_feature_config: Union[TextOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        super().__init__(output_feature_config, output_features, **kwargs)

    @classmethod
    def get_output_dtype(cls):
        return torch.int32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.decoder_obj.config.max_sequence_length])

    def update_metrics(
        self,
        targets: Tensor,
        predictions: Dict[str, Tensor],
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        """Updates metrics with the given targets and predictions.

        If decoded_targets and decoded_predictions are provided, as through LLM model types, then additional
        response-based metrics like BLEU and ROUGE are also computed.

        Args:
            targets: Tensor with target values for this output feature.
            predictions: Dict of tensors returned by predictions().
        """
        if tokenizer is not None:
            # Decode the targets and predictions to compute response-based metrics using the initialized tokenizer.
            decoded_targets, decoded_predictions = get_decoded_targets_and_predictions(targets, predictions, tokenizer)

        for metric_name, metric_fn in self._metric_functions.items():
            prediction_key = get_metric_tensor_input(metric_name)
            try:
                if prediction_key == RESPONSE:
                    if tokenizer is not None:
                        # RESPONSE metrics cannot be computed if decoded texts are not provided.
                        # Decoded texts are only provided using the LLM model type.
                        if decoded_targets is not None and decoded_predictions is not None:
                            # Move metric function to the device of the predictions.
                            # For CUDA, it can be computed on any of the GPUs since it uses allgather to collect
                            # the results from all GPUs and compute the final metric.
                            # We use 'predictions' as the key since it is always present in the predictions dict.
                            device = "cuda" if predictions["predictions"].is_cuda else "cpu"
                            metric_fn = metric_fn.to(device)
                            if metric_name == "bleu":
                                # BLEU takes in targets as a list.
                                metric_fn.update(decoded_predictions, [decoded_targets])
                            else:
                                metric_fn.update(decoded_predictions, decoded_targets)
                else:
                    metric_fn = metric_fn.to(predictions[prediction_key].device)
                    metric_fn.update(predictions[prediction_key].detach(), targets)
            except Exception as e:
                logger.info(f"Ran into error when calculating metric {metric_name}. Skipping. The error is: {e}")

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.decoder.vocab_size = feature_metadata["vocab_size"]
        feature_config.decoder.max_sequence_length = feature_metadata["max_sequence_length"]
        if isinstance(feature_config.loss.class_weights, (list, tuple)):
            # [0, 0] for UNK and PAD
            feature_config.loss.class_weights = [0, 0] + feature_config.loss.class_weights
            if len(feature_config.loss.class_weights) != feature_config.decoder.vocab_size:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({})".format(
                        len(feature_config.loss.class_weights), feature_config.decoder.vocab_size
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
            if feature_config.class_similarities:
                distances = feature_config.class_similarities
                temperature = feature_config.loss.class_similarities_temperature
                for i in range(len(distances)):
                    distances[i, :] = softmax(distances[i, :], temperature=temperature)
                feature_config.loss.class_similarities = distances
            else:
                raise ValueError(
                    "class_similarities_temperature > 0,"
                    "but no class similarities are provided "
                    "for feature {}".format(feature_config.column)
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

        tokenizer = None
        if metadata["preprocessing"]["tokenizer"] == "hf_tokenizer":
            tokenizer = get_tokenizer(
                metadata["preprocessing"]["tokenizer"],
                metadata["preprocessing"]["vocab_file"],
                metadata["preprocessing"]["pretrained_model_name_or_path"],
            )

        if predictions_col in result:
            token_col = result[predictions_col]

            def idx2str(pred):
                if tokenizer is None:
                    return [
                        metadata["idx2str"][token] if token < len(metadata["idx2str"]) else UNKNOWN_SYMBOL
                        for token in pred
                    ]
                return tokenizer.tokenizer.batch_decode(pred, skip_special_tokens=True)

            result[predictions_col] = token_col.map(idx2str)

            # Add additional response column that represents the predicted text output
            # as a single string instead of a list of tokens.
            def idx2response(pred):
                if tokenizer is None:
                    # This works because we treat each word as a token.
                    return " ".join(
                        [
                            metadata["idx2str"][token] if token < len(metadata["idx2str"]) else UNKNOWN_SYMBOL
                            for token in pred
                        ]
                    )
                return tokenizer.tokenizer.batch_decode([pred], skip_special_tokens=True)

            result[f"{self.feature_name}_response"] = token_col.map(idx2response)

        last_preds_col = f"{self.feature_name}_{LAST_PREDICTIONS}"
        if last_preds_col in result:

            def last_idx2str(last_pred):
                if last_pred < len(metadata["idx2str"]):
                    return metadata["idx2str"][last_pred]
                return UNKNOWN_SYMBOL

            result[last_preds_col] = result[last_preds_col].map(last_idx2str)

        probs_col = f"{self.feature_name}_{PROBABILITIES}"
        prob_col = f"{self.feature_name}_{PROBABILITY}"

        # "Summarizes" the `result`'s probability-related output:
        # - result[probs_col]:
        #       Each row is now a list of "max" probabilities. Each element is the probability of the argmax token for
        #       the given time step.
        #
        #       Note that we intentionally do not return full list of probabilties for each time step because the output
        #       of postprocess_predictions is saved to disk and the full probability distribution can be huge,
        #       especially for large vocab sizes:
        #           dataset_size x sequence_length x vocab_size
        #
        #       TODO: Add a mechanism that lets the user save the full probability distribution if they want.
        # - result[prob_col]:
        #       Each row is the overall probability of the sequence. This is the product of the max probabilities over
        #       all time steps.
        if probs_col in result:
            # result[probs_col]: From PredictModule, each row has a list of size (sequence_length) of a list of
            # probabiltiies of (vocab_size). compute_token_probabilities gets the maximum probability per timestep.
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
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SequencePostprocessing(metadata)

    @staticmethod
    def get_schema_cls():
        return TextOutputFeatureConfig
