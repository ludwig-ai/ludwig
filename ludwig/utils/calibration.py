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


class TemperatureScaling(nn.Module):
    """Implements temperature scaling of logits. Based on results from On Calibration of Modern Neural Networks:
    https://arxiv.org/abs/1706.04599.

    Implementation inspired by https://github.com/gpleiss/temperature_scaling
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False).to(self.device)

    def calibrate(self, logits, labels):
        """Calibrate."""
        logits = torch.tensor(logits, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)
        self.temperature.requires_grad = True
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        logging.info(
            "Before temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (before_temperature_nll, before_temperature_ece)
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.forward(logits), labels).item()
        after_temperature_ece = ece_criterion(self.forward(logits), labels).item()
        logging.info("Optimal temperature: %.3f" % self.temperature.item())
        logging.info(
            "After temperature scaling:\n"
            "    Negative log-likelihood: %.3f\n"
            "    Expected Calibration Error: %.3f" % (after_temperature_nll, after_temperature_ece)
        )
        self.temperature.requires_grad = False

    def forward(self, logits: torch.Tensor):
        return torch.div(logits, self.temperature)
