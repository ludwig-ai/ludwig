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

import torch
import torch.nn as nn


class ECELoss(nn.Module):
    """Calculates the Expected Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins: int = 15):
        """n_bins (int): number of confidence interval bins."""
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, one_hot_labels):
        softmaxes = nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        labels = torch.argmax(one_hot_labels, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class CalibrationModule(nn.Module, ABC):
    @abstractmethod
    def calibrate(self, logits, labels):
        """Calibrate output probabilities using logits and labels from validation set."""
        return NotImplementedError()


class TemperatureScaling(CalibrationModule):
    """Implements temperature scaling of logits. Based on results from On Calibration of Modern Neural Networks:
    https://arxiv.org/abs/1706.04599. Temperature scaling scales all logits by the same constant factor, so though
    it may modify output probabilities it will never change argmax or top-n predictions. In the case of binary
    classification with a threshold, however, calibration may change predictions.

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

    def calibrate(self, logits, labels):
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
        before_temperature_nll = nll_criterion(logits, one_hot_labels).item()
        before_temperature_ece = ece_criterion(logits, one_hot_labels).item()
        logging.info(
            "Before temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (before_temperature_nll, before_temperature_ece)
        )

        # Optimizes the temperature to minimize NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.001, max_iter=1000, line_search_fn="strong_wolfe")

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.scale_logits(logits), one_hot_labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.scale_logits(logits), one_hot_labels).item()
        after_temperature_ece = ece_criterion(self.scale_logits(logits), one_hot_labels).item()
        logging.info("Optimal temperature: %.3f" % self.temperature.item())
        logging.info(
            "After temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (after_temperature_nll, after_temperature_ece)
        )
        self.temperature.requires_grad = False
        # This should never happen, but if expected calibration error is higher after optimizing temperature, revert.
        if after_temperature_ece > before_temperature_ece:
            logging.warning(
                "Expected calibration error higher after scaling, "
                "reverting to temperature=%.3f." % original_temperature.item()
            )
            with torch.no_grad():
                self.temperature.data = original_temperature.data

    def scale_logits(self, logits: torch.Tensor):
        return torch.div(logits, self.temperature)

    def forward(self, logits: torch.Tensor):
        """Converts logits to probabilities."""
        scaled_logits = self.scale_logits(logits)
        if self.binary:
            return torch.sigmoid(scaled_logits)
        else:
            return torch.softmax(scaled_logits, -1)


class MatrixScaling(CalibrationModule):
    """Implements matrix scaling of logits. Only use this with a large dataset, matrix scaling has a tendency to
    overfit small dataset. Also, unlike temperature scaling, matrix scaling can change the argmax or top-n
    predictions.

    Args:
    num_classes: The number of classes.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.w = nn.Parameter(torch.eye(self.num_classes), requires_grad=False).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False).to(self.device)

    def calibrate(self, logits, labels):
        logits = torch.as_tensor(logits, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)
        one_hot_labels = nn.functional.one_hot(labels, self.num_classes).float()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)
        self.w.requires_grad = True
        self.b.requires_grad = True
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, one_hot_labels).item()
        before_temperature_ece = ece_criterion(logits, one_hot_labels).item()
        logging.info(
            "Before matrix scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (before_temperature_nll, before_temperature_ece)
        )

        # Optimizes the temperature to minimize NLL
        optimizer = torch.optim.LBFGS([self.w, self.b], lr=0.001, max_iter=1000, line_search_fn="strong_wolfe")

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.scale_logits(logits), one_hot_labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.scale_logits(logits), one_hot_labels).item()
        after_temperature_ece = ece_criterion(self.scale_logits(logits), one_hot_labels).item()
        logging.info(
            "After matrix scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (after_temperature_nll, after_temperature_ece)
        )
        self.w.requires_grad = False
        self.b.requires_grad = False
        # This should never happen, but if expected calibration error is higher after optimizing temperature, revert.
        if after_temperature_ece > before_temperature_ece:
            logging.warning("Expected calibration error higher after matrix scaling, reverting to identity.")
            with torch.no_grad():
                self.w.data = torch.eye(self.num_classes)
                self.b.data = torch.zeros(self.num_classes)

    def scale_logits(self, logits: torch.Tensor):
        return torch.matmul(logits, self.w) + self.b

    def forward(self, logits: torch.Tensor):
        """Converts logits to probabilities."""
        return torch.softmax(self.scale_logits(logits), -1)
