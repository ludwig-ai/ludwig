from whylogs.core.resolvers import Resolver
from whylogs.core.datatypes import DataType, Fractional, Integral, String
from typing import Dict, List
from whylogs.core.metrics import StandardMetric
from whylogs.core.metrics.metrics import Metric, OperationResult, MetricConfig
from whylogs.core.metrics.metric_components import FractionalComponent
from whylogs.core.preprocessing import PreprocessedColumn
from typing import Any
from dataclasses import dataclass
from whylogs.core.configs import SummaryConfig
from ludwig.utils.image_utils import is_image_score
from ludwig.utils.audio_utils import is_audio_score


@dataclass(frozen=True)
class IsImageMetric(Metric):
    score: FractionalComponent
    name = "ludwig_metric"

    @property
    def namespace(self) -> str:
        return "is_image"

    def columnar_update(self, view: PreprocessedColumn) -> OperationResult:
        successes = 0
        if view.pandas.strings is not None:
            self.score.set(is_image_score(None, view.pandas.strings.to_list()[0], column=""))
            successes += len(view.pandas.strings)
        if view.list.strings:
            successes += len(view.list.strings)

        failures = 0
        if view.list.objs:
            failures = len(view.list.objs)
        return OperationResult(successes=successes, failures=failures)

    def to_summary_dict(self, cfg: SummaryConfig) -> Dict[str, Any]:
        return {"image_score": self.score.value}

    @classmethod
    def zero(cls, config: MetricConfig) -> "IsImageMetric":
        return IsImageMetric(score=FractionalComponent(0.0))


@dataclass(frozen=True)
class IsAudioMetric(Metric):
    score: FractionalComponent
    name = "ludwig_metric"

    @property
    def namespace(self) -> str:
        return "is_audio"

    def columnar_update(self, view: PreprocessedColumn) -> OperationResult:
        successes = 0
        if view.pandas.strings is not None:
            self.score.set(is_audio_score(view.pandas.strings.to_list()[0]))
            successes += len(view.pandas.strings)
        if view.list.strings:
            successes += len(view.list.strings)

        failures = 0
        if view.list.objs:
            failures = len(view.list.objs)
        return OperationResult(successes=successes, failures=failures)

    def to_summary_dict(self, cfg: SummaryConfig) -> Dict[str, Any]:
        return {"audio_score": self.score.value}

    @classmethod
    def zero(cls, config: MetricConfig) -> "IsAudioMetric":
        return IsAudioMetric(score=FractionalComponent(0.0))


class LudwigWhyResolver(Resolver):
    """Default whylogs resolver with additional metrics for the String type to support Ludwig type inference for image
    and audio columns.
    """

    def resolve(self, name: str, why_type: DataType, column_schema) -> Dict[str, Metric]:
        metrics: List[StandardMetric] = [StandardMetric.counts, StandardMetric.types]

        if isinstance(why_type, Integral):
            metrics.append(StandardMetric.distribution)
            metrics.append(StandardMetric.ints)
            metrics.append(StandardMetric.cardinality)
            metrics.append(StandardMetric.frequent_items)
        elif isinstance(why_type, Fractional):
            metrics.append(StandardMetric.cardinality)
            metrics.append(StandardMetric.distribution)
        elif isinstance(why_type, String):  # Catch all category as we map 'object' here
            metrics.append(StandardMetric.cardinality)
            metrics.append(IsImageMetric)
            metrics.append(IsAudioMetric)

            # TODO: Average words metric.

            # if column_schema.cfg.track_unicode_ranges:
            #     metrics.append(StandardMetric.unicode_range)
            metrics.append(StandardMetric.unicode_range)

            metrics.append(StandardMetric.distribution)  # 'object' columns can contain Decimal
            metrics.append(StandardMetric.frequent_items)

        if column_schema.cfg.fi_disabled:
            metrics.remove(StandardMetric.frequent_items)

        result: Dict[str, Metric] = {}
        for m in metrics:
            result[m.name] = m.zero(column_schema.cfg)
        return result
