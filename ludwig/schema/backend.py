from typing import Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BackendConfig(schema_utils.BaseMarshmallowConfig):
    """Global backend compute resource/usage configuration."""

    type: str = schema_utils.StringOptions(
        options=["local", "ray", "horovod"],
        default="local",
        description='How the job will be distributed, one of "local", "ray", or "horovod".',
    )

    cache_dir: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Where the preprocessed data will be written on disk, defaults to the location of the "
        "input dataset.",
    )

    cache_credentials: dict = schema_utils.Dict(
        default=None,
        allow_none=True,
        description="Optional dictionary of credentials (or path to credential JSON file) used to write to the cache.",
    )

    processor: Optional["ProcessorConfig"]  # noqa: F821

    trainer: Optional["TrainerConfig"]  # noqa: F821

    loader: Optional["LoaderConfig"]  # noqa: F821


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


@DeveloperAPI
@ludwig_dataclass
class TrainerConfig(schema_utils.BaseMarshmallowConfig):
    """Configuration for distributed training (only supported by the `ray` backend)."""

    type: str = schema_utils.StringOptions(
        options=["horovod"], default="horovod", allow_none=True, description="The distributed training backend to use."
    )

    use_gpu: bool = schema_utils.Boolean(
        default=True,
        allow_none=True,
        description="Whether to use GPUs for training (defaults to true when the cluster has at least one GPU).",
    )

    num_workers: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="How many Horovod workers to use for training (defaults to the number of GPUs, or 1 if no GPUs "
        "are found).",
    )

    resources_per_worker = schema_utils.Dict(
        default=None,
        allow_none=True,
        description="The Ray resources to assign to each Horovod worker (defaults to 1 CPU and 1 GPU if available).",
    )

    logdir: str = schema_utils.String(
        default=None, allow_none=True, description="Path to the file directory where logs should be persisted."
    )

    max_retries: int = schema_utils.NonNegativeInteger(
        default=3, description="Number of retries when Ray actors fail (defaults to 3)."
    )


@DeveloperAPI
@ludwig_dataclass
class LoaderConfig(schema_utils.BaseMarshmallowConfig):
    """Configuration for the "last mile" data ingest from processed data to tensor batches used for training the
    model."""

    fully_executed: bool = schema_utils.Boolean(
        default=True,
        description="Force full evaluation of the preprocessed dataset by loading all blocks into cluster memory / "
        "storage (defaults to true). Disable this if the dataset is much larger than the total amount of "
        "cluster memory allocated to the Ray object store and you notice that object spilling is "
        "occurring frequently during training.",
    )

    window_size_bytes: Union[int, str] = schema_utils.OneOfOptionsField(
        default=None,
        field_options=[schema_utils.PositiveInteger(), schema_utils.String(default="auto")],
        allow_none=True,
        description="Load and shuffle the preprocessed dataset in discrete windows of this size (defaults to null, "
        "meaning data will not be windowed). Try configuring this is if shuffling is taking a very long "
        "time, indicated by every epoch of training taking many minutes to start. In general, larger "
        "window sizes result in more uniform shuffling (which can lead to better model performance in "
        "some cases), while smaller window sizes will be faster to load. This setting is particularly "
        "useful when running hyperopt over a large dataset.",
    )
