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

from ludwig.constants import COLUMN, HIDDEN, LOGITS, NAME, PREDICTIONS, PROBABILITIES, PROC_COLUMN, SET
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.schema.features.set_feature import SetInputFeatureConfig, SetOutputFeatureConfig
from ludwig.types import (
    FeatureMetadataDict,
    FeaturePostProcessingOutputDict,
    PreprocessingConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils import output_feature_utils
from ludwig.utils.strings_utils import create_vocabulary, UNKNOWN_SYMBOL
from ludwig.utils.tokenizers import get_tokenizer_from_registry, TORCHSCRIPT_COMPATIBLE_TOKENIZERS
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class _SetPreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by SetFeatureMixin.add_feature_data.

    If is_bag is true, forward returns a vector for each sample indicating counts of each token. Else, forward returns a
    multi-hot vector for each sample indicating presence of each token.
    """

    def __init__(self, metadata: TrainingSetMetadataDict, is_bag: bool = False):
        super().__init__()
        if metadata["preprocessing"]["tokenizer"] not in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
            raise ValueError(
                f"{metadata['preprocessing']['tokenizer']} is not supported by torchscript. Please use "
                f"one of {TORCHSCRIPT_COMPATIBLE_TOKENIZERS}."
            )

        self.lowercase = metadata["preprocessing"]["lowercase"]
        self.tokenizer = get_tokenizer_from_registry(metadata["preprocessing"]["tokenizer"])()
        self.vocab_size = metadata["vocab_size"]
        self.unknown_symbol = UNKNOWN_SYMBOL
        self.unit_to_id = metadata["str2idx"]
        self.is_bag = is_bag

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        """Takes a list of strings and returns a tensor of counts for each token."""
        if not torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        if self.lowercase:
            sequences = [sequence.lower() for sequence in v]
        else:
            sequences = v

        unit_sequences = self.tokenizer(sequences)
        # refines type of unit_sequences from Any to List[List[str]]
        assert torch.jit.isinstance(unit_sequences, List[List[str]]), "unit_sequences is not a list of lists."

        set_matrix = torch.zeros(len(unit_sequences), self.vocab_size, dtype=torch.float32)
        for sample_idx, unit_sequence in enumerate(unit_sequences):
            sequence_length = len(unit_sequence)
            for i in range(sequence_length):
                curr_unit = unit_sequence[i]
                if curr_unit in self.unit_to_id:
                    curr_id = self.unit_to_id[curr_unit]
                else:
                    curr_id = self.unit_to_id[self.unknown_symbol]

                if self.is_bag:
                    set_matrix[sample_idx][curr_id] += 1
                else:
                    set_matrix[sample_idx][curr_id] = 1

        return set_matrix


class _SetPostprocessing(torch.nn.Module):
    """Torchscript-enabled version of postprocessing done by SetFeatureMixin.add_feature_data."""

    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.idx2str = {i: v for i, v in enumerate(metadata["idx2str"])}
        self.predictions_key = PREDICTIONS
        self.probabilities_key = PROBABILITIES
        self.unk = UNKNOWN_SYMBOL

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> FeaturePostProcessingOutputDict:
        predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)
        probabilities = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.probabilities_key)

        inv_preds: List[List[str]] = []
        filtered_probs: List[torch.Tensor] = []
        for sample_idx, sample in enumerate(predictions):
            sample_preds: List[str] = []
            pos_sample_idxs: List[int] = []
            pos_class_idxs: List[int] = []
            for class_idx, is_positive in enumerate(sample):
                if is_positive == 1:
                    sample_preds.append(self.idx2str.get(class_idx, self.unk))
                    pos_sample_idxs.append(sample_idx)
                    pos_class_idxs.append(class_idx)
            inv_preds.append(sample_preds)
            filtered_probs.append(probabilities[pos_sample_idxs, pos_class_idxs])

        return {
            self.predictions_key: inv_preds,
            self.probabilities_key: filtered_probs,
        }


class _SetPredict(PredictModule):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)
        probabilities = torch.sigmoid(logits)

        predictions = torch.greater_equal(probabilities, self.threshold)
        predictions = predictions.type(torch.int64)

        return {self.predictions_key: predictions, self.probabilities_key: probabilities, self.logits_key: logits}


class SetFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return SET

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
            num_most_frequent=preprocessing_parameters["most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            add_special_symbols=False,
            processor=backend.df_engine,
        )
        return {
            "idx2str": vocabulary.vocab,
            "str2idx": vocabulary.str2idx,
            "str2freq": vocabulary.str2freq,
            "vocab_size": len(vocabulary.str2idx),
            "max_set_size": vocabulary.line_length_max,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters: PreprocessingConfigDict, backend):
        def to_dense(x):
            feature_vector = set_str_to_idx(x, metadata["str2idx"], preprocessing_parameters["tokenizer"])

            set_vector = np.zeros((len(metadata["str2idx"]),))
            set_vector[feature_vector] = 1
            return set_vector.astype(np.bool_)

        return backend.df_engine.map_objects(column, to_dense)

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
        proc_df[feature_config[PROC_COLUMN]] = SetFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]],
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


class SetInputFeature(SetFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: SetInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.bool, torch.int64, torch.float32]

        encoder_output = self.encoder_obj(inputs)

        return encoder_output

    @property
    def input_dtype(self):
        return torch.bool

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([len(self.encoder_obj.config.vocab)])

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.vocab = feature_metadata["idx2str"]

    @staticmethod
    def get_schema_cls():
        return SetInputFeatureConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SetPreprocessing(metadata)


class SetOutputFeature(SetFeatureMixin, OutputFeature):
    def __init__(
        self,
        output_feature_config: Union[SetOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        self.threshold = output_feature_config.threshold
        super().__init__(output_feature_config, output_features, **kwargs)
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def metric_kwargs(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}

    def create_predict_module(self) -> PredictModule:
        return _SetPredict(self.threshold)

    def get_prediction_set(self):
        return {PREDICTIONS, PROBABILITIES, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return torch.bool

    @property
    def input_shape(self) -> torch.Size:
        return self.decoder_obj.input_shape

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.decoder_obj.config.num_classes])

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.decoder.num_classes = feature_metadata["vocab_size"]
        if isinstance(feature_config.loss.class_weights, (list, tuple)):
            if len(feature_config.loss.class_weights) != feature_config.decoder.num_classes:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({}) for feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and their order and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        len(feature_config.loss.class_weights),
                        feature_config.decoder.num_classes,
                        feature_config.name,
                    )
                )

        if isinstance(feature_config.loss.class_weights, dict):
            if feature_metadata["str2idx"].keys() != feature_config.loss.class_weights.keys():
                raise ValueError(
                    "The class_weights keys ({}) are not compatible with "
                    "the classes ({}) of feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        feature_config.loss.class_weights.keys(),
                        feature_metadata["str2idx"].keys(),
                        feature_config.name,
                    )
                )
            else:
                class_weights = feature_config.loss.class_weights
                idx2str = feature_metadata["idx2str"]
                class_weights_list = [class_weights[s] for s in idx2str]
                feature_config.loss.class_weights = class_weights_list

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

            def idx2str(pred_set):
                return [metadata["idx2str"][i] for i, pred in enumerate(pred_set) if pred]

            result[predictions_col] = result[predictions_col].map(idx2str)

        probabilities_col = f"{self.feature_name}_{PROBABILITIES}"
        if probabilities_col in result:

            def get_prob(prob_set):
                # Cast to float32 because empty np.array objects are np.float64, causing mismatch errors during saving.
                return np.array([prob for prob in prob_set if prob >= self.threshold], dtype=np.float32)

            result[probabilities_col] = result[probabilities_col].map(get_prob)

        return result

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _SetPostprocessing(metadata)

    @staticmethod
    def get_schema_cls():
        return SetOutputFeatureConfig
