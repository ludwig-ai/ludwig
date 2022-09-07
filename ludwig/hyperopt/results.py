# !/usr/bin/env python

from dataclasses import dataclass
from typing import Any, List

from dataclasses_json import dataclass_json

try:
    from ray.tune import ExperimentAnalysis
except ImportError:
    ExperimentAnalysis = Any


@dataclass_json
@dataclass
class TrialResults:
    parameters: dict
    metric_score: float
    training_stats: dict
    eval_stats: dict


@dataclass
class HyperoptResults:
    ordered_trials: List[TrialResults]
    experiment_analysis: ExperimentAnalysis
