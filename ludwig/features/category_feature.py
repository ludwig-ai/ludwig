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
    CATEGORY,
    CATEGORY_DISTRIBUTION,
    COLUMN,
    HIDDEN,
    LOGITS,
    NAME,
    PREDICTIONS,
    PREPROCESSING,
    PROBABILITIES,
    PROBABILITY,
    PROC_COLUMN,
    PROJECTION_INPUT,
)
from ludwig.error import InputDataError
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.features.vector_feature import VectorFeatureMixin
from ludwig.schema.features.category_feature import (
    CategoryDistributionOutputFeatureConfig,
    CategoryInputFeatureConfig,
    CategoryOutputFeatureConfig,
)
from ludwig.schema.features.loss.loss import CORNLossConfig
from ludwig.types import (
    FeatureMetadataDict,
    FeaturePostProcessingOutputDict,
    PreprocessingConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils import calibration, output_feature_utils
from ludwig.utils.eval_utils import ConfusionMatrix
from ludwig.utils.math_utils import int_type, softmax
from ludwig.utils.strings_utils import create_vocabulary_single_token, UNKNOWN_SYMBOL
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class _CategoryPreprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.str2idx = metadata["str2idx"]
        if UNKNOWN_SYMBOL in self.str2idx:
            self.unk = self.str2idx[UNKNOWN_SYMBOL]
        else:
            # self.unk is set to 0 to comply with Torchscript type tracing and will
            # likely not be used during training, but potentially during inference
            self.unk = 0

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if not torch.jit.isinstance(v, List[str]):
            raise ValueError(f"Unsupported input: {v}")

        indices = [self.str2idx.get(s.strip(), self.unk) for s in v]
        return torch.tensor(indices, dtype=torch.int32)


class _CategoryPostprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.idx2str = {i: v for i, v in enumerate(metadata["idx2str"])}
        self.unk = UNKNOWN_SYMBOL
        self.predictions_key = PREDICTIONS
        self.probabilities_key = PROBABILITIES

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> FeaturePostProcessingOutputDict:
        predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)
        probabilities = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.probabilities_key)

        inv_preds = [self.idx2str.get(pred, self.unk) for pred in predictions]

        return {
            self.predictions_key: inv_preds,
            self.probabilities_key: probabilities,
        }


class _CategoryPredict(PredictModule):
    def __init__(self, calibration_module=None, use_cumulative_probs=False):
        super().__init__()
        self.calibration_module = calibration_module

        # Derive the label from the cumulative probability distribution of the ordered category logits.
        # Taken from CORN loss implementation:
        # https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/dataset.py#L123
        self.use_cumulative_probs = use_cumulative_probs

    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)

        if self.use_cumulative_probs:
            if self.calibration_module is not None:
                probabilities = self.calibration_module(logits)
            else:
                probabilities = torch.sigmoid(logits)
            probabilities = torch.cumprod(probabilities, dim=1)

            predict_levels = probabilities > 0.5
            predictions = torch.sum(predict_levels, dim=1)
        else:
            if self.calibration_module is not None:
                probabilities = self.calibration_module(logits)
            else:
                probabilities = torch.softmax(logits, -1)
            predictions = torch.argmax(probabilities, -1)

        predictions = predictions.long()

        # EXPECTED SHAPE OF RETURNED TENSORS
        # predictions: [batch_size]
        # probabilities: [batch_size, num_classes]
        # logits: [batch_size, num_classes]
        return {self.predictions_key: predictions, self.probabilities_key: probabilities, self.logits_key: logits}


class CategoryFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return CATEGORY

    @staticmethod
    def cast_column(column, backend):
        return column.astype(str)

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        idx2str, str2idx, str2freq = create_vocabulary_single_token(
            column,
            num_most_frequent=preprocessing_parameters["most_common"],
            processor=backend.df_engine,
        )

        if "vocab" in preprocessing_parameters and preprocessing_parameters["vocab"]:  # Check that vocab is non-empty
            # If vocab was explciitly provided, override the inferred vocab
            idx2str = preprocessing_parameters["vocab"]
            str2idx = {s: i for i, s in enumerate(idx2str)}
            str2freq = {k: str2freq.get(k, 0) for k in idx2str}

        if "fallback_label" in preprocessing_parameters:
            # This is a category output feature for LLMs
            # Check if the fallback label is in the vocab, if not add it.
            if preprocessing_parameters["fallback_label"] not in str2idx:
                str2idx[preprocessing_parameters["fallback_label"]] = len(str2idx)
                idx2str.append(preprocessing_parameters["fallback_label"])
                str2freq[preprocessing_parameters["fallback_label"]] = 0

        vocab_size = len(str2idx)
        if not is_input_feature and vocab_size <= 1:
            # Category output feature with vocab size 1
            raise InputDataError(
                column.name,
                CATEGORY,
                f"""
                At least 2 distinct values are required for category output features, but column
                only contains {str(idx2str)}.
                """,
            )
        if vocab_size <= 1:
            # Category input feature with vocab size 1
            logger.info(
                f"Input feature '{column.name}' contains only 1 distinct value {str(idx2str)}. This is not useful"
                " for machine learning models because this feature has zero variance. Consider removing this feature"
                " from your input features."
            )
        return {"idx2str": idx2str, "str2idx": str2idx, "str2freq": str2freq, "vocab_size": vocab_size}

    @staticmethod
    def feature_data(backend, column, metadata):
        def __replace_token_with_idx(value: Any, metadata: TrainingSetMetadataDict, fallback_symbol_idx: int) -> int:
            stripped_value = value.strip()
            if stripped_value in metadata["str2idx"]:
                return metadata["str2idx"][stripped_value]
            logger.warning(
                f"""
                Encountered unknown symbol '{stripped_value}' for '{column.name}' during category
                feature preprocessing. This should never happen during training. If this happens during
                inference, this may be an indication that not all possible symbols were present in your
                training set. Consider re-splitting your data to ensure full representation, or setting
                preprocessing.most_common parameter to be smaller than this feature's total vocabulary
                size, {len(metadata["str2idx"])}, which will ensure that the model is architected and
                trained with an UNKNOWN symbol. Returning the index for the most frequent symbol,
                {metadata["idx2str"][fallback_symbol_idx]}, instead.
                """
            )
            return fallback_symbol_idx

        # No unknown symbol in Metadata from preprocessing means that all values
        # should be mappable to vocabulary
        if UNKNOWN_SYMBOL not in metadata["str2idx"]:
            # If no unknown is defined, just use the most popular token's index as the fallback index
            most_popular_token = max(metadata["str2freq"], key=metadata["str2freq"].get)
            most_popular_token_idx = metadata["str2idx"].get(most_popular_token)
            return backend.df_engine.map_objects(
                column,
                lambda x: __replace_token_with_idx(x, metadata, most_popular_token_idx),
                meta=(column.name, int),
            ).astype(int_type(metadata["vocab_size"]))
        else:
            return backend.df_engine.map_objects(
                column,
                lambda x: (
                    metadata["str2idx"][x.strip()]
                    if x.strip() in metadata["str2idx"]
                    else metadata["str2idx"][UNKNOWN_SYMBOL]
                ),
                meta=(column.name, int),
            ).astype(int_type(metadata["vocab_size"]))

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
        proc_df[feature_config[PROC_COLUMN]] = CategoryFeatureMixin.feature_data(
            backend,
            input_df[feature_config[COLUMN]],
            metadata[feature_config[NAME]],
        )

        return proc_df


class CategoryDistributionFeatureMixin(VectorFeatureMixin):
    @staticmethod
    def type():
        return CATEGORY_DISTRIBUTION

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        idx2str = preprocessing_parameters["vocab"]
        str2idx = {s: i for i, s in enumerate(idx2str)}
        return {
            "preprocessing": preprocessing_parameters,
            "idx2str": idx2str,
            "str2idx": str2idx,
            "vocab_size": len(idx2str),
        }


class CategoryInputFeature(CategoryFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: CategoryInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        assert len(inputs.shape) == 1 or (len(inputs.shape) == 2 and inputs.shape[1] == 1)

        inputs = inputs.reshape(-1, 1)
        if inputs.dtype == torch.int8 or inputs.dtype == torch.int16:
            inputs = inputs.type(torch.int)
        encoder_output = self.encoder_obj(inputs)

        return encoder_output

    @property
    def input_dtype(self):
        return torch.int32

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self.encoder_obj.output_shape)

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.vocab = feature_metadata["idx2str"]
        feature_config.encoder.skip = feature_metadata[PREPROCESSING].get("cache_encoder_embeddings", False)

    @staticmethod
    def get_schema_cls():
        return CategoryInputFeatureConfig

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _CategoryPreprocessing(metadata)


class CategoryOutputFeature(CategoryFeatureMixin, OutputFeature):
    def __init__(
        self,
        output_feature_config: Union[CategoryOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        self.num_classes = output_feature_config.num_classes
        self.top_k = output_feature_config.top_k

        # TODO(travis): make this more general to other cumulative loss functions
        self.use_cumulative_probs = isinstance(output_feature_config.loss, CORNLossConfig)

        super().__init__(output_feature_config, output_features, **kwargs)
        if hasattr(output_feature_config.decoder, "num_classes"):
            output_feature_config.decoder.num_classes = output_feature_config.num_classes
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]

        # EXPECTED SHAPES FOR RETURNED TENSORS
        # logits: shape [batch_size, num_classes]
        # hidden: shape [batch_size, size of final fully connected layer]
        return {LOGITS: self.decoder_obj(hidden), PROJECTION_INPUT: hidden}

    def create_calibration_module(self, feature: CategoryOutputFeatureConfig) -> torch.nn.Module:
        """Creates the appropriate calibration module based on the feature config.

        Today, only one type of calibration ("temperature_scaling") is available, but more options may be supported in
        the future.
        """
        if feature.calibration:
            calibration_cls = calibration.get_calibration_cls(CATEGORY, "temperature_scaling")
            return calibration_cls(num_classes=self.num_classes)
        return None

    def create_predict_module(self) -> PredictModule:
        return _CategoryPredict(
            calibration_module=self.calibration_module, use_cumulative_probs=self.use_cumulative_probs
        )

    def get_prediction_set(self):
        return {PREDICTIONS, PROBABILITIES, LOGITS}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @classmethod
    def get_output_dtype(cls):
        return torch.int64

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    def metric_kwargs(self):
        return {"top_k": self.top_k, "num_classes": self.num_classes}

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.num_classes = feature_metadata["vocab_size"]
        feature_config.top_k = min(feature_config.num_classes, feature_config.top_k)

        # If labels are provided, then this is a classification task for LLMs
        if hasattr(feature_config.preprocessing, "vocab"):
            # Enrich the feature config's decoder with str2idx
            feature_config.decoder.str2idx = feature_metadata["str2idx"]

        if isinstance(feature_config.loss.class_weights, (list, tuple)):
            if len(feature_config.loss.class_weights) != feature_config.num_classes:
                raise ValueError(
                    "The length of class_weights ({}) is not compatible with "
                    "the number of classes ({}) for feature {}. "
                    "Check the metadata JSON file to see the classes "
                    "and their order and consider there needs to be a weight "
                    "for the <UNK> class too.".format(
                        len(feature_config.loss.class_weights),
                        feature_config.num_classes,
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

                if all_rows_length != feature_config.num_classes:
                    raise ValueError(
                        "The size of the class_similarities matrix of {} is "
                        "{}, different from the number of classes ({}). "
                        "Check the metadata JSON file to see the classes "
                        "and their order and "
                        "consider <UNK> class too.".format(
                            feature_config.column, all_rows_length, feature_config.num_classes
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
        overall_stats = {}
        confusion_matrix = ConfusionMatrix(targets, predictions[PREDICTIONS], labels=train_set_metadata["idx2str"])
        overall_stats["confusion_matrix"] = confusion_matrix.cm.tolist()
        overall_stats["overall_stats"] = confusion_matrix.stats()
        overall_stats["per_class_stats"] = confusion_matrix.per_class_stats()

        return overall_stats

    def postprocess_predictions(
        self,
        predictions,
        metadata,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in predictions:
            if "idx2str" in metadata:
                predictions[predictions_col] = predictions[predictions_col].map(lambda pred: metadata["idx2str"][pred])

        probabilities_col = f"{self.feature_name}_{PROBABILITIES}"
        if probabilities_col in predictions:
            prob_col = f"{self.feature_name}_{PROBABILITY}"
            predictions[prob_col] = predictions[probabilities_col].map(max)
            predictions[probabilities_col] = predictions[probabilities_col].map(lambda pred: pred.tolist())
            if "idx2str" in metadata:
                for i, label in enumerate(metadata["idx2str"]):
                    key = f"{probabilities_col}_{label}"

                    # Use default param to force a capture before the loop completes, see:
                    # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                    predictions[key] = predictions[probabilities_col].map(
                        lambda prob, i=i: prob[i],
                    )

        top_k_col = f"{self.feature_name}_predictions_top_k"
        if top_k_col in predictions:
            if "idx2str" in metadata:
                predictions[top_k_col] = predictions[top_k_col].map(
                    lambda pred_top_k: [metadata["idx2str"][pred] for pred in pred_top_k]
                )

        return predictions

    @staticmethod
    def get_schema_cls():
        return CategoryOutputFeatureConfig

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _CategoryPostprocessing(metadata)


class CategoryDistributionOutputFeature(CategoryDistributionFeatureMixin, CategoryOutputFeature):
    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.num_classes])

    @staticmethod
    def get_schema_cls():
        return CategoryDistributionOutputFeatureConfig
