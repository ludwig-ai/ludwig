# !/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass
from typing import Any, Dict, List

from dataclasses_json import dataclass_json

try:
    from ray.tune import ExperimentAnalysis
except ImportError:
    ExperimentAnalysis = Any


@dataclass_json
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
