# !/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from ray.tune import ExperimentAnalysis
except ImportError:
    ExperimentAnalysis = Any


@dataclass
class TrialResults:
    parameters: Dict[str, Any]
    metric_score: float
    training_stats: Dict
    eval_stats: Dict


@dataclass
class HyperoptResults:
    ordered_trials: List[TrialResults]


@dataclass
class RayTuneResults(HyperoptResults):
    experiment_analysis: ExperimentAnalysis
