from typing import Any, Dict

from whylogs.core import DatasetSchema
from whylogs.core.metrics.metrics import MetricConfig
from whylogs.core.resolvers import Resolver

from ludwig.profiling.why_resolver import LudwigWhyResolver


class LudwigWhySchema(DatasetSchema):
    types: Dict[str, Any] = {}
    default_configs: MetricConfig = MetricConfig()
    resolvers: Resolver = LudwigWhyResolver()
    cache_size: int = 1024
    schema_based_automerge: bool = False
