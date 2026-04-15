# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
"""Anomaly detection output feature using Deep One-Class Classification methods.

This module provides the AnomalyOutputFeature, which implements three hypersphere-based
anomaly detection objectives:

- Deep SVDD (default): Trains a neural network to map all normal inputs inside a compact
  hypersphere. The squared distance ||z - c||^2 from the encoder output z to the center c
  is the anomaly score. Geometric and interpretable; works well for homogeneous normal data.
  Reference: Ruff et al., "Deep One-Class Classification", ICML 2018.

- Deep SAD: Semi-supervised extension of SVDD. Normal examples are pulled toward c while
  confirmed anomaly examples (target=1) are pushed away. Useful when a small number of
  labeled anomalies are available at training time.
  Reference: Ruff et al., "Deep Semi-Supervised Anomaly Detection", ICLR 2020.

- DROCC: Prevents collapse (all representations mapping to c) by adding an adversarial
  perturbation term. Recommended when using expressive encoders such as transformers.
  Reference: Goyal et al., "DROCC: Deep Robust One-Class Classification", ICML 2020.

Multimodal anomaly detection works natively through Ludwig's ECD architecture: add any
combination of input features (text, image, audio, tabular) and the combiner will fuse
them before the anomaly decoder.

Center initialization:
    The hypersphere center c must be initialized after the first epoch by calling
    ``output_feature.initialize_center(center_tensor)``. This is typically handled
    automatically by the Ludwig trainer when an anomaly output feature is detected.
    The center is set to the mean of all encoder outputs on the training set.
"""

import logging

import numpy as np
import torch

from ludwig.constants import (
    ANOMALY,
    ANOMALY_SCORE,
    COLUMN,
    HIDDEN,
    NAME,
    PREDICTIONS,
    PROC_COLUMN,
)
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature, PredictModule
from ludwig.schema.features.anomaly_feature import AnomalyOutputFeatureConfig
from ludwig.types import (
    FeatureConfigDict,
    FeatureMetadataDict,
    FeaturePostProcessingOutputDict,
    ModelConfigDict,
    PreprocessingConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils import output_feature_utils
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)


class _AnomalyPredict(PredictModule):
    """Converts raw anomaly scores to predictions (is_anomaly bool) using a threshold.

    Args:
        threshold: Float threshold above which anomaly scores are classified as anomalous.
            Use ``float('inf')`` to disable hard classification (only anomaly_score is returned).
    """

    def __init__(self, threshold: float = float("inf")):
        super().__init__()
        self.threshold = threshold
        self.anomaly_score_key = ANOMALY_SCORE

    def forward(self, inputs: dict[str, torch.Tensor], feature_name: str) -> dict[str, torch.Tensor]:
        anomaly_score = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.anomaly_score_key)
        # Threshold-based binary classification.
        predictions = anomaly_score >= self.threshold
        return {
            self.anomaly_score_key: anomaly_score,
            self.predictions_key: predictions,
        }


class AnomalyFeatureMixin(BaseFeatureMixin):
    """Mixin providing preprocessing utilities for anomaly output features.

    Anomaly detection is typically unsupervised — the target column contains 0 (normal) or 1 (anomaly) labels for
    evaluation, or -1 for unlabeled examples (used by Deep SAD). If no target column is present in the dataset, the
    feature runs in fully unsupervised mode.
    """

    @staticmethod
    def type() -> str:
        return ANOMALY

    @staticmethod
    def cast_column(column: DataFrame, backend) -> DataFrame:
        """Cast the target anomaly label column to float32.

        Labels should be 0 (normal), 1 (anomaly), or -1 (unlabeled for Deep SAD). If the column contains NaN values
        (unlabeled), they are filled with -1.
        """
        return column.fillna(-1).astype(np.float32)

    @staticmethod
    def get_feature_meta(
        config: ModelConfigDict,
        column: DataFrame,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        is_input_feature: bool,
    ) -> FeatureMetadataDict:
        return {}

    @staticmethod
    def add_feature_data(
        feature_config: FeatureConfigDict,
        input_df: DataFrame,
        proc_df: dict[str, DataFrame],
        metadata: TrainingSetMetadataDict,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input: bool,
    ) -> None:
        column_name = feature_config[COLUMN]
        if column_name in input_df.columns:
            # Labels present: cast to float32 (0=normal, 1=anomaly, -1=unlabeled).
            col = input_df[column_name].fillna(-1).astype(np.float32)
        else:
            # Fully unsupervised: fill with -1 (unlabeled) so the loss still works.
            logger.info(
                f"Anomaly feature '{feature_config[NAME]}': no target column found in dataset. "
                "Running in fully unsupervised mode (all labels set to -1)."
            )
            col = backend.df_engine.from_pandas(
                np.full(len(input_df), -1.0, dtype=np.float32),
            )

        proc_df[feature_config[PROC_COLUMN]] = col
        return proc_df


class AnomalyOutputFeature(AnomalyFeatureMixin, OutputFeature):
    """Output feature for anomaly detection using Deep One-Class Classification.

    This feature maps input representations to an anomaly score via a hypersphere decoder.
    The anomaly score is the squared distance from the encoder output to a learned center c:

        anomaly_score = ||encoder_output - c||^2

    Training objectives (controlled by the ``loss.type`` config field):
        - ``deep_svdd``: Hard or soft boundary SVDD. Minimizes mean(||z - c||^2) for all points.
          For soft boundary, a radius R is estimated as the nu-th quantile of distances.
        - ``deep_sad``: Semi-supervised. Normal/unlabeled points pulled toward c; labeled anomalies
          (target=1) pushed away from c via inverted distance.
        - ``drocc``: Adds adversarial perturbations to prevent collapse. Recommended for
          expressive encoders.

    Center initialization:
        After the first epoch (or at any time), call ``initialize_center(center)`` to set the
        hypersphere center c. The trainer does this automatically for anomaly output features.
        Until initialization, the center defaults to zeros.

    Predictions:
        - ``anomaly_score``: Float scalar per sample (primary output for ranking and metrics).
        - ``predictions``: Bool per sample — True if anomaly_score >= threshold.

    Threshold:
        Configure via the ``threshold`` field in the output feature config:
        - ``"auto"``: Automatically select the threshold as the ``threshold_percentile``-th
          percentile of validation-set anomaly scores after training.
        - A float value: use as a fixed decision boundary.

    Multimodal:
        Works with any ECD input features. The combiner fuses inputs before the anomaly decoder.
    """

    def __init__(
        self,
        output_feature_config: AnomalyOutputFeatureConfig | dict,
        output_features: dict[str, OutputFeature],
        **kwargs,
    ):
        self._threshold = output_feature_config.threshold
        self._threshold_percentile = getattr(output_feature_config, "threshold_percentile", 95.0)
        super().__init__(output_feature_config, output_features, **kwargs)
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    @staticmethod
    def type() -> str:
        return ANOMALY

    def initialize_center(self, center: torch.Tensor) -> None:
        """Initialize the hypersphere center c from the mean of first-epoch encoder outputs.

        Delegates to the underlying AnomalyDecoder. Call this after collecting encoder outputs
        for the entire training set at the end of epoch 0.

        Args:
            center: Tensor of shape ``[latent_dim]`` — typically ``encoder_outputs.mean(dim=0)``.
        """
        self.decoder_obj.initialize_center(center)

    @property
    def center_initialized(self) -> bool:
        """Whether the hypersphere center has been initialized."""
        return getattr(self.decoder_obj, "_center_initialized", False)

    def logits(self, inputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Compute anomaly scores from hidden representations.

        Args:
            inputs: Dict with at least key ``HIDDEN`` — the decoder input.

        Returns:
            Dict with ``anomaly_score`` key containing per-sample distances ``||z - c||^2``.
        """
        hidden = inputs[HIDDEN]
        anomaly_score = self.decoder_obj(hidden)
        return {ANOMALY_SCORE: anomaly_score}

    def create_predict_module(self) -> PredictModule:
        # Resolve numeric threshold; "auto" starts as inf and is updated after training.
        if isinstance(self._threshold, str) and self._threshold == "auto":
            threshold = float("inf")
        else:
            threshold = float(self._threshold)
        return _AnomalyPredict(threshold=threshold)

    def get_prediction_set(self) -> set[str]:
        return {ANOMALY_SCORE, PREDICTIONS}

    @classmethod
    def get_output_dtype(cls) -> torch.dtype:
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    def train_loss(self, targets: torch.Tensor, predictions: dict[str, torch.Tensor], feature_name: str):
        """Compute the anomaly detection training loss.

        Args:
            targets: Ground-truth labels (0=normal, 1=anomaly, -1=unlabeled). Shape [batch].
            predictions: Dict of tensors including ``anomaly_score``.
            feature_name: Feature name used to look up the correct tensor.

        Returns:
            Scalar loss tensor.
        """
        prediction_key = output_feature_utils.get_feature_concat_name(
            feature_name, type(self.train_loss_function).get_loss_inputs()
        )
        return self.train_loss_function(predictions[prediction_key], targets)

    def eval_loss(self, targets: torch.Tensor, predictions: dict[str, torch.Tensor]):
        prediction_key = type(self.train_loss_function).get_loss_inputs()
        return self.eval_loss_metric.get_current_value(predictions[prediction_key].detach(), targets)

    def update_metrics(self, targets: torch.Tensor, predictions: dict[str, torch.Tensor]) -> None:
        """Update evaluation metrics with current batch.

        Args:
            targets: Ground-truth labels (0=normal, 1=anomaly, -1=unlabeled). Shape [batch].
            predictions: Dict of tensors including ``anomaly_score``.
        """
        from ludwig.modules.metric_registry import get_metric_tensor_input

        for metric_name, metric_fn in self._metric_functions.items():
            prediction_key = get_metric_tensor_input(metric_name)
            metric_fn = metric_fn.to(predictions[prediction_key].device)
            metric_fn.update(predictions[prediction_key].detach(), targets)

    def auto_set_threshold(self, validation_anomaly_scores: torch.Tensor) -> None:
        """Set the anomaly threshold automatically from validation-set anomaly scores.

        After training, call this with the anomaly scores from the validation set to
        automatically determine the decision threshold. The threshold is set to the
        ``threshold_percentile``-th percentile of validation scores.

        Args:
            validation_anomaly_scores: 1D tensor of anomaly scores from the validation set.
        """
        if isinstance(self._threshold, str) and self._threshold == "auto":
            threshold_value = float(torch.quantile(validation_anomaly_scores, self._threshold_percentile / 100.0))
            logger.info(
                f"Anomaly feature '{self.feature_name}': auto-selected threshold "
                f"{threshold_value:.6f} ({self._threshold_percentile}th percentile of validation scores)."
            )
            self._threshold = threshold_value
            # Update the predict module's threshold too.
            if hasattr(self, "_prediction_module") and self._prediction_module is not None:
                self._prediction_module.module.threshold = threshold_value

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        """Calculate overall anomaly detection statistics.

        Computes AUROC and F1-max over all thresholds when ground-truth labels are available.
        If only unlabeled data is available (all targets == -1), returns an empty dict.

        Args:
            predictions: Dict with keys ``anomaly_score`` and ``predictions``.
            targets: 1D array of ground-truth labels (0, 1, or -1).
            train_set_metadata: Unused metadata.

        Returns:
            Dict of overall statistics.
        """
        overall_stats = {}
        scores = np.array(predictions.get(ANOMALY_SCORE, []))
        preds = np.array(predictions.get(PREDICTIONS, []))
        targets = np.array(targets)

        # Only compute label-dependent stats if labeled data is available.
        labeled_mask = targets >= 0
        if labeled_mask.sum() < 2:
            logger.info("Anomaly feature: no labeled validation data available; skipping AUROC/F1 stats.")
            return overall_stats

        labeled_scores = scores[labeled_mask]
        labeled_targets = targets[labeled_mask]
        labeled_preds = preds[labeled_mask]

        if len(np.unique(labeled_targets)) < 2:
            logger.info("Anomaly feature: only one class in labeled data; skipping AUROC/F1 stats.")
            return overall_stats

        try:
            from sklearn.metrics import roc_auc_score

            overall_stats["roc_auc"] = float(roc_auc_score(labeled_targets, labeled_scores))
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")

        try:
            from sklearn.metrics import f1_score, precision_score, recall_score

            overall_stats["f1"] = float(f1_score(labeled_targets, labeled_preds, zero_division=0))
            overall_stats["precision"] = float(precision_score(labeled_targets, labeled_preds, zero_division=0))
            overall_stats["recall"] = float(recall_score(labeled_targets, labeled_preds, zero_division=0))
        except Exception as e:
            logger.warning(f"Could not compute F1/precision/recall: {e}")

        # F1-max: best F1 over all thresholds.
        try:
            from sklearn.metrics import f1_score as sk_f1

            best_f1 = 0.0
            best_threshold = None
            for t in np.unique(labeled_scores):
                preds_t = (labeled_scores >= t).astype(int)
                f1 = sk_f1(labeled_targets, preds_t, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(t)
            overall_stats["f1_max"] = best_f1
            overall_stats["f1_max_threshold"] = best_threshold
        except Exception as e:
            logger.warning(f"Could not compute F1-max: {e}")

        return overall_stats

    def postprocess_predictions(
        self,
        result,
        metadata,
    ) -> FeaturePostProcessingOutputDict:
        """Post-process model output columns for saving.

        Renames ``anomaly_score`` and ``predictions`` columns with the feature name prefix.

        Args:
            result: DataFrame with model output columns.
            metadata: Feature metadata (unused for anomaly).

        Returns:
            Modified result DataFrame.
        """
        # Anomaly score is already float — no further post-processing needed.
        return result

    @staticmethod
    def get_schema_cls():
        return AnomalyOutputFeatureConfig

    @classmethod
    def get_postproc_output_dtype(cls, metadata: TrainingSetMetadataDict) -> str:
        return "float32"
