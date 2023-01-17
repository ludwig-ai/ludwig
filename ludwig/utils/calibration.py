#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type, Union

import numpy as np
import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY
from ludwig.utils.registry import DEFAULT_KEYS, Registry

logger = logging.getLogger(__name__)

calibration_registry = Registry()


@DeveloperAPI
def register_calibration(name: str, features: Union[str, List[str]], default=False):
    """Registers a calibration implementation for a list of features."""
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = calibration_registry.get(feature, {})
            feature_registry[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    feature_registry[key] = cls
            calibration_registry[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
def get_calibration_cls(feature: str, calibration_method: str) -> Type["CalibrationModule"]:
    """Get calibration class for specified feature type and calibration method."""
    if not calibration_method:
        return None
    if feature in calibration_registry:
        if calibration_method in calibration_registry[feature]:
            return calibration_registry[feature][calibration_method]
        else:
            raise ValueError(f"Calibration method {calibration_method} not supported for {feature} output features")
    else:
        raise ValueError(f"Calibration not yet supported for {feature} output features")
    return None


@DeveloperAPI
class ECELoss(nn.Module):
    """Calculates the Expected Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return an average of the gaps, weighted by the number of samples in each bin.

    References:
        Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. 2015.

        Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
        "On Calibration of Modern Neural Networks." PMLR 2017.
    """

    def __init__(self, n_bins: int = 15):
        """n_bins (int): number of confidence interval bins."""
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.Tensor, one_hot_labels: torch.Tensor) -> torch.Tensor:
        softmaxes = nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        labels = torch.argmax(one_hot_labels, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculates |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


@DeveloperAPI
@dataclass
class CalibrationResult:
    """Tracks results of probability calibration."""

    before_calibration_nll: float
    before_calibration_ece: float
    after_calibration_nll: float
    after_calibration_ece: float


@DeveloperAPI
class CalibrationModule(nn.Module, ABC):
    @abstractmethod
    def train_calibration(
        self, logits: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]
    ) -> CalibrationResult:
        """Calibrate output probabilities using logits and labels from validation set."""
        return NotImplementedError()


@DeveloperAPI
@register_calibration("temperature_scaling", [BINARY, CATEGORY], default=True)
class TemperatureScaling(CalibrationModule):
    """Implements temperature scaling of logits. Based on results from "On Calibration of Modern Neural Networks":
    https://arxiv.org/abs/1706.04599. Temperature scaling scales all logits by the same constant factor. Though it
    may modify output probabilities it will never change argmax or categorical top-n predictions. In the case of
    binary classification with a threshold, however, calibration may change predictions.

    Implementation inspired by https://github.com/gpleiss/temperature_scaling

    Args:
        num_classes: The number of classes. Must be 2 if binary is True.
        binary: If binary is true, logits is expected to be a 1-dimensional array. If false, logits is a 2-dimensional
                array of shape (num_examples, num_classes).
    """

    def __init__(self, num_classes: int = 2, binary: bool = False):
        super().__init__()
        self.num_classes = 2 if binary else num_classes
        self.binary = binary
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False).to(self.device)

    def train_calibration(
        self, logits: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]
    ) -> CalibrationResult:
        logits = torch.as_tensor(logits, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)
        one_hot_labels = nn.functional.one_hot(labels, self.num_classes).float()
        if self.binary:
            # Treat binary classification as multi-class with 2 classes to re-use code.
            # The math works out the same: softmax([0, a])[1] == sigmoid(a)
            logits = torch.stack([torch.zeros_like(logits), logits], axis=-1)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)
        # Saves the original temperature parameter, in case something goes wrong in optimization.
        original_temperature = self.temperature.clone().detach()
        self.temperature.requires_grad = True
        # Calculate NLL and ECE before temperature scaling
        before_calibration_nll = nll_criterion(logits, one_hot_labels).item()
        before_calibration_ece = ece_criterion(logits, one_hot_labels).item()
        logger.info(
            "Before temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (before_calibration_nll, before_calibration_ece)
        )

        # Optimizes the temperature to minimize NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.scale_logits(logits), one_hot_labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_calibration_nll = nll_criterion(self.scale_logits(logits), one_hot_labels).item()
        after_calibration_ece = ece_criterion(self.scale_logits(logits), one_hot_labels).item()
        logger.info("Optimal temperature: %.3f" % self.temperature.item())
        logger.info(
            "After temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (after_calibration_nll, after_calibration_ece)
        )
        self.temperature.requires_grad = False
        # This should never happen, but if expected calibration error is higher after optimizing temperature, revert.
        if after_calibration_ece > before_calibration_ece:
            logger.warning(
                "Expected calibration error higher after scaling, "
                "reverting to temperature=%.3f." % original_temperature.item()
            )
            with torch.no_grad():
                self.temperature.data = original_temperature.data
        return CalibrationResult(
            before_calibration_nll, before_calibration_ece, after_calibration_nll, after_calibration_ece
        )

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.div(logits, self.temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities."""
        scaled_logits = self.scale_logits(logits)
        if self.binary:
            return torch.sigmoid(scaled_logits)
        else:
            return torch.softmax(scaled_logits, -1)


@DeveloperAPI
@register_calibration("matrix_scaling", CATEGORY, default=False)
class MatrixScaling(CalibrationModule):
    """Implements matrix scaling of logits, as described in Beyond temperature scaling: Obtaining well-calibrated
    multiclass probabilities with Dirichlet calibration https://arxiv.org/abs/1910.12656.

    Unlike temperature scaling which has only one free parameter, matrix scaling has n_classes x (n_classes + 1)
    parameters. Use this only with a large validation set, as matrix scaling has a tendency to overfit small datasets.
    Also, unlike temperature scaling, matrix scaling can change the argmax or top-n predictions.

    NOTE: Matrix Scaling is not exposed in the UI or config yet, though it may be in a future release after testing.

    Args:
    num_classes: The number of classes.
    off_diagonal_l2: The regularization weight for off-diagonal matrix entries.
    mu: The regularization weight for bias vector. Defaults to off_diagonal_l2 if not specified.
    """

    def __init__(self, num_classes: int = 2, off_diagonal_l2: float = 0.01, mu: float = None):
        super().__init__()
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.w = nn.Parameter(torch.eye(self.num_classes), requires_grad=False).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False).to(self.device)
        self.off_diagonal_l2 = off_diagonal_l2
        self.mu = off_diagonal_l2 if mu is None else mu

    def train_calibration(
        self, logits: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]
    ) -> CalibrationResult:
        logits = torch.as_tensor(logits, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)
        one_hot_labels = nn.functional.one_hot(labels, self.num_classes).float()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)
        self.w.requires_grad = True
        self.b.requires_grad = True
        # Calculate NLL and ECE before temperature scaling
        before_calibration_nll = nll_criterion(logits, one_hot_labels).item()
        before_calibration_ece = ece_criterion(logits, one_hot_labels).item()
        logger.info(
            "Before matrix scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (before_calibration_nll, before_calibration_ece)
        )

        # Optimizes the linear transform to minimize NLL
        optimizer = torch.optim.LBFGS([self.w, self.b], lr=0.001, max_iter=200, line_search_fn="strong_wolfe")

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.scale_logits(logits), one_hot_labels) + self.regularization_terms()
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after matrix scaling
        after_calibration_nll = nll_criterion(self.scale_logits(logits), one_hot_labels).item()
        after_calibration_ece = ece_criterion(self.scale_logits(logits), one_hot_labels).item()
        logger.info(
            "After matrix scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (after_calibration_nll, after_calibration_ece)
        )
        self.w.requires_grad = False
        self.b.requires_grad = False
        # This should never happen, but if expected calibration error is higher after optimizing matrix, revert.
        if after_calibration_ece > before_calibration_ece:
            logger.warning("Expected calibration error higher after matrix scaling, reverting to identity.")
            with torch.no_grad():
                self.w.data = torch.eye(self.num_classes)
                self.b.data = torch.zeros(self.num_classes)
        return CalibrationResult(
            before_calibration_nll, before_calibration_ece, after_calibration_nll, after_calibration_ece
        )

    def regularization_terms(self) -> torch.Tensor:
        """Off-Diagonal and Intercept Regularisation (ODIR).

        Described in "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet
        calibration" https://proceedings.neurips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf
        """
        off_diagonal_entries = torch.masked_select(self.w, ~torch.eye(self.num_classes, dtype=bool))
        weight_matrix_loss = self.off_diagonal_l2 * torch.linalg.vector_norm(off_diagonal_entries)
        bias_vector_loss = self.mu * torch.linalg.vector_norm(self.b, 2)
        return bias_vector_loss + weight_matrix_loss

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.w, logits.T).T + self.b

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities."""
        return torch.softmax(self.scale_logits(logits), -1)
