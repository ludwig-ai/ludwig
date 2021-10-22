#! /usr/bin/env python
# coding=utf-8
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


from ludwig.constants import *
from ludwig.decoders.generic_decoders import Regressor
from ludwig.encoders.binary_encoders import ENCODER_REGISTRY
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.modules.loss_modules import BWCEWLoss
from ludwig.modules.metric_modules import Accuracy, BWCEWLMetric, ROCAUCMetric
from ludwig.utils.eval_utils import ConfusionMatrix, average_precision_score,\
    precision_recall_curve, roc_auc_score, roc_curve
from ludwig.utils.misc_utils import set_default_value, set_default_values
from ludwig.utils import strings_utils

logger = logging.getLogger(__name__)


class BinaryFeatureMixin:
    type = BINARY
    preprocessing_defaults = {
        "missing_value_strategy": FILL_WITH_CONST,
        "fill_value": 0,
    }

    fill_value_schema = {
        "anyOf": [
            {"type": "integer", "minimum": 0, "maximum": 1},
            {"type": "string", "enum": strings_utils.all_bool_strs()},
        ]
    }

    preprocessing_schema = {
        "missing_value_strategy": {
            "type": "string",
            "enum": MISSING_VALUE_STRATEGY_OPTIONS,
        },
        "fill_value": fill_value_schema,
        "computed_fill_value": fill_value_schema,
    }

    @staticmethod
    def cast_column(column, backend):
        # todo maybe move code from add_feature_data here
        #  + figure out what NaN is in a bool column
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        if column.dtype != object:
            return {}

        distinct_values = backend.df_engine.compute(column.drop_duplicates())
        if len(distinct_values) > 2:
            raise ValueError(
                f"Binary feature column {column.name} expects 2 distinct values, "
                f"found: {distinct_values.values.tolist()}"
            )

        str2bool = {v: strings_utils.str2bool(v) for v in distinct_values}
        bool2str = [
            k for k, v in sorted(str2bool.items(), key=lambda item: item[1])
        ]

        return {
            "str2bool": str2bool,
            "bool2str": bool2str,
        }

    @staticmethod
    def add_feature_data(
        feature,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters,
        backend,
        skip_save_processed_input,
    ):
        column = input_df[feature[COLUMN]]

        if column.dtype == object:
            metadata = metadata[feature[NAME]]
            if "str2bool" in metadata:
                column = column.map(lambda x: metadata["str2bool"][x])
            else:
                # No predefined mapping from string to bool, so compute it directly
                column = column.map(strings_utils.str2bool)

        proc_df[feature[PROC_COLUMN]] = column.astype(np.bool_).values
        return proc_df


class BinaryInputFeature(BinaryFeatureMixin, InputFeature):
    encoder = "passthrough"
    norm = None
    dropout = False

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs, training=None, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.bool, torch.int64, torch.float32]
        assert len(inputs.shape) in [1, 2]
        inputs_exp = inputs[:, None]
        encoder_outputs = self.encoder_obj(
            inputs_exp)
        return encoder_outputs

    @property
    def input_dtype(self):
        return torch.bool

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(
            input_feature, feature_metadata, *args, **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = ENCODER_REGISTRY


class BinaryOutputFeature(BinaryFeatureMixin, OutputFeature):
    decoder = "regressor"
    loss = {TYPE: SOFTMAX_CROSS_ENTROPY}
    metric_functions = {LOSS: None, ACCURACY: None}
    default_validation_metric = ACCURACY
    threshold = 0.5

    def __init__(self, feature):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def predictions(self, inputs, **kwargs):  # hidden
        logits = inputs[LOGITS]
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= self.threshold
        return {
            PROBABILITIES: probabilities,
            PREDICTIONS: predictions,
            LOGITS: logits,
        }

    def _setup_loss(self):
        self.train_loss_function = BWCEWLoss(
            positive_class_weight=self.loss["positive_class_weight"],
            robust_lambda=self.loss["robust_lambda"],
            confidence_penalty=self.loss["confidence_penalty"],
        )
        self.eval_loss_function = self.train_loss_function

    def _setup_metrics(self):
        self.metric_functions = {}  # needed to shadow class variable
        self.metric_functions[LOSS] = BWCEWLMetric(
            positive_class_weight=self.loss["positive_class_weight"],
            robust_lambda=self.loss["robust_lambda"],
            confidence_penalty=self.loss["confidence_penalty"],
        )
        self.metric_functions[ACCURACY] = Accuracy()
        self.metric_functions[ROC_AUC] = ROCAUCMetric()

    def get_prediction_set(self):
        return {PREDICTIONS, PROBABILITIES, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return torch.bool

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @staticmethod
    def update_config_with_metadata(
        input_feature, feature_metadata, *args, **kwargs
    ):
        pass

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        overall_stats = {}
        confusion_matrix = ConfusionMatrix(
            targets, predictions[PREDICTIONS], labels=["False", "True"]
        )
        overall_stats["confusion_matrix"] = confusion_matrix.cm.tolist()
        overall_stats["overall_stats"] = confusion_matrix.stats()
        overall_stats["per_class_stats"] = confusion_matrix.per_class_stats()
        fpr, tpr, thresholds = roc_curve(targets, predictions[PROBABILITIES])
        overall_stats["roc_curve"] = {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
        }
        overall_stats["roc_auc_macro"] = roc_auc_score(
            targets, predictions[PROBABILITIES], average="macro"
        )
        overall_stats["roc_auc_micro"] = roc_auc_score(
            targets, predictions[PROBABILITIES], average="micro"
        )
        ps, rs, thresholds = precision_recall_curve(
            targets, predictions[PROBABILITIES]
        )
        overall_stats["precision_recall_curve"] = {
            "precisions": ps.tolist(),
            "recalls": rs.tolist(),
        }
        overall_stats["average_precision_macro"] = average_precision_score(
            targets, predictions[PROBABILITIES], average="macro"
        )
        overall_stats["average_precision_micro"] = average_precision_score(
            targets, predictions[PROBABILITIES], average="micro"
        )
        overall_stats["average_precision_samples"] = average_precision_score(
            targets, predictions[PROBABILITIES], average="samples"
        )

        return overall_stats

    def postprocess_predictions(
        self,
        result,
        metadata,
        output_directory,
        backend,
    ):
        class_names = ["False", "True"]
        if "bool2str" in metadata:
            class_names = metadata["bool2str"]

        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            if "bool2str" in metadata:
                result[predictions_col] = backend.df_engine.map_objects(
                    result[predictions_col],
                    lambda pred: metadata["bool2str"][pred],
                )

        probabilities_col = f"{self.feature_name}_{PROBABILITIES}"
        if probabilities_col in result:
            false_col = f"{probabilities_col}_{class_names[0]}"
            result[false_col] = backend.df_engine.map_objects(
                result[probabilities_col], lambda prob: 1 - prob
            )

            true_col = f"{probabilities_col}_{class_names[1]}"
            result[true_col] = result[probabilities_col]

            prob_col = f"{self.feature_name}_{PROBABILITY}"
            result[prob_col] = result[[false_col, true_col]].max(axis=1)

            result[probabilities_col] = backend.df_engine.map_objects(
                result[probabilities_col], lambda prob: [1 - prob, prob]
            )

        return result

    @staticmethod
    def populate_defaults(output_feature):
        # If Loss is not defined, set an empty dictionary
        set_default_value(output_feature, LOSS, {})
        set_default_values(
            output_feature[LOSS],
            {
                "robust_lambda": 0,
                "confidence_penalty": 0,
                "positive_class_weight": None,
                "weight": 1,
            },
        )

        set_default_value(output_feature[LOSS], "robust_lambda", 0)
        set_default_value(output_feature[LOSS], "confidence_penalty", 0)
        set_default_value(output_feature[LOSS], "positive_class_weight", None)
        set_default_value(output_feature[LOSS], "weight", 1)

        set_default_values(
            output_feature,
            {
                "threshold": 0.5,
                "dependencies": [],
                "reduce_input": SUM,
                "reduce_dependencies": SUM,
            },
        )

    decoder_registry = {
        "regressor": Regressor,
        "null": Regressor,
        "none": Regressor,
        "None": Regressor,
        None: Regressor,
    }
