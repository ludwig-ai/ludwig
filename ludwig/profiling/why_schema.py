from typing import Dict, Any

from whylogs.core.resolvers import Resolver
from whylogs.core.metrics.metrics import MetricConfig
from whylogs.core import DatasetSchema

from ludwig.profiling.why_resolver import LudwigWhyResolver


class LudwigWhySchema(DatasetSchema):
    types: Dict[str, Any] = {}
    default_configs: MetricConfig = MetricConfig()
    resolvers: Resolver = LudwigWhyResolver()
    cache_size: int = 1024
    schema_based_automerge: bool = False
