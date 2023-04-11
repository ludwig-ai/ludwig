from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ProcessorConfig(schema_utils.BaseMarshmallowConfig):
    """Configuration for distributed data processing (only supported by the `ray` backend)."""

    type: str = schema_utils.StringOptions(
        options=["dask", "modin"],
        default="dask",
        description='Distributed data processing engine to use. `"dask"`: (default) a lazily executed version of '
        'distributed Pandas. `"modin"`: an eagerly executed version of distributed Pandas.',
    )

    parallelism: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="(dask only) The number of partitions to divide the dataset into (defaults to letting Dask figure "
        "this out automatically).",
    )

    persist: bool = schema_utils.Boolean(
        default=True,
        allow_none=True,
        description="(dask only) Whether intermediate stages of preprocessing should be cached in distributed memory.",
    )
