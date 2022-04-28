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
    HIDDEN,
    JACCARD,
    LOGITS,
    LOSS,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NAME,
    PREDICTIONS,
    PROBABILITIES,
    PROC_COLUMN,
    SET,
    SIGMOID_CROSS_ENTROPY,
    SUM,
    TIED,
    TYPE,
)
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.utils import output_feature_utils
from ludwig.utils.misc_utils import get_from_registry, set_default_value
from ludwig.utils.strings_utils import create_vocabulary, tokenizer_registry, UNKNOWN_SYMBOL
from ludwig.utils.tokenizers import TORCHSCRIPT_COMPATIBLE_TOKENIZERS

logger = logging.getLogger(__name__)


class _SetPreprocessing(torch.nn.Module):
    """Torchscript-enabled version of preprocessing done by SetFeatureMixin.add_feature_data.

    If is_bag is true, forward returns a vector for each sample indicating counts of each token. Else, forward returns a
    multi-hot vector for each sample indicating presence of each token.
    """

    def __init__(self, metadata: Dict[str, Any], is_bag: bool = False):
        super().__init__()
        if metadata["preprocessing"]["tokenizer"] not in TORCHSCRIPT_COMPATIBLE_TOKENIZERS:
            raise ValueError(
                f"{metadata['preprocessing']['tokenizer']} is not supported by torchscript. Please use "
                f"one of {TORCHSCRIPT_COMPATIBLE_TOKENIZERS}."
            )

        self.lowercase = metadata["preprocessing"]["lowercase"]
        self.tokenizer = get_from_registry(metadata["preprocessing"]["tokenizer"], tokenizer_registry)()
        self.vocab_size = metadata["vocab_size"]
        self.unknown_symbol = UNKNOWN_SYMBOL
        self.unit_to_id = metadata["str2idx"]
        self.is_bag = is_bag

    def forward(self, v: Union[List[str], List[torch.Tensor], torch.Tensor]):
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
    def preprocessing_defaults():
        return {
            "tokenizer": "space",
            "most_common": 10000,
            "lowercase": False,
            "missing_value_strategy": FILL_WITH_CONST,
            "fill_value": UNKNOWN_SYMBOL,
        }

    @staticmethod
    def preprocessing_schema():
        return {
            "tokenizer": {"type": "string", "enum": sorted(list(tokenizer_registry.keys()))},
            "most_common": {"type": "integer", "minimum": 0},
            "lowercase": {"type": "boolean"},
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
        idx2str, str2idx, str2freq, max_size, _, _, _, _ = create_vocabulary(
            column,
            preprocessing_parameters["tokenizer"],
            num_most_frequent=preprocessing_parameters["most_common"],
            lowercase=preprocessing_parameters["lowercase"],
            add_special_symbols=False,
            processor=backend.df_engine,
        )
        return {
            "idx2str": idx2str,
            "str2idx": str2idx,
            "str2freq": str2freq,
            "vocab_size": len(str2idx),
            "max_set_size": max_size,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, backend):
        def to_dense(x):
            feature_vector = set_str_to_idx(x, metadata["str2idx"], preprocessing_parameters["tokenizer"])

            set_vector = np.zeros((len(metadata["str2idx"]),))
            set_vector[feature_vector] = 1
            return set_vector.astype(np.bool)

        return backend.df_engine.map_objects(column, to_dense)

    @staticmethod
    def add_feature_data(
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        proc_df[feature_config[PROC_COLUMN]] = SetFeatureMixin.feature_data(
            input_df[feature_config[COLUMN]].astype(str),
            metadata[feature_config[NAME]],
            preprocessing_parameters,
            backend,
        )
        return proc_df


class SetInputFeature(SetFeatureMixin, InputFeature):
    encoder = "embed"
    vocab = []

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.bool, torch.int64]

        encoder_output = self.encoder_obj(inputs)

        return {"encoder_output": encoder_output}

    @property
    def input_dtype(self):
        return torch.bool

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([len(self.vocab)])

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        input_feature["vocab"] = feature_metadata["idx2str"]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _SetPreprocessing(metadata)


class SetOutputFeature(SetFeatureMixin, OutputFeature):
    decoder = "classifier"
    loss = {TYPE: SIGMOID_CROSS_ENTROPY}
    metric_functions = {LOSS: None, JACCARD: None}
    default_validation_metric = JACCARD
    num_classes = 0
    threshold = 0.5

    def __init__(self, feature, output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def loss_kwargs(self):
        return self.loss

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
        return torch.Size([self.num_classes])

    @staticmethod
    def update_config_with_metadata(output_feature, feature_metadata, *args, **kwargs):
        output_feature["num_classes"] = feature_metadata["vocab_size"]
        if isinstance(output_feature[LOSS]["class_weights"], (list, tuple)):
            if len(output_feature[LOSS]["class_weights"]) != output_feature["num_classes"]:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({}) for feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and their order and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        len(output_feature[LOSS]["class_weights"]), output_feature["num_classes"], output_feature[NAME]
                    )
                )

        if isinstance(output_feature[LOSS]["class_weights"], dict):
            if feature_metadata["str2idx"].keys() != output_feature[LOSS]["class_weights"].keys():
                raise ValueError(
                    "The class_weights keys ({}) are not compatible with "
                    "the classes ({}) of feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and consider there needs to be a weight "
                    "for the <UNK> and <PAD> class too.".format(
                        output_feature[LOSS]["class_weights"].keys(),
                        feature_metadata["str2idx"].keys(),
                        output_feature[NAME],
                    )
                )
            else:
                class_weights = output_feature[LOSS]["class_weights"]
                idx2str = feature_metadata["idx2str"]
                class_weights_list = [class_weights[s] for s in idx2str]
                output_feature[LOSS]["class_weights"] = class_weights_list

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
        self,
        result,
        metadata,
        output_directory,
        backend,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:

            def idx2str(pred_set):
                return [metadata["idx2str"][i] for i, pred in enumerate(pred_set) if pred]

            result[predictions_col] = backend.df_engine.map_objects(
                result[predictions_col],
                idx2str,
            )

        probabilities_col = f"{self.feature_name}_{PROBABILITIES}"
        if probabilities_col in result:
            threshold = self.threshold

            def get_prob(prob_set):
                return [prob for prob in prob_set if prob >= threshold]

            result[probabilities_col] = backend.df_engine.map_objects(
                result[probabilities_col],
                get_prob,
            )

        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {TYPE: SIGMOID_CROSS_ENTROPY, "weight": 1})
        set_default_value(output_feature[LOSS], "weight", 1)
        set_default_value(output_feature[LOSS], "class_weights", None)

        set_default_value(output_feature, "threshold", 0.5)
        set_default_value(output_feature, "dependencies", [])
        set_default_value(output_feature, "reduce_input", SUM)
        set_default_value(output_feature, "reduce_dependencies", SUM)
