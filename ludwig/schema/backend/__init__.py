from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseBackendConfig(schema_utils.BaseMarshmallowConfig):
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


@DeveloperAPI
@ludwig_dataclass
class RayBackendConfig(BaseBackendConfig):
    type: str = schema_utils.ProtectedString("ray", description="Distribute training with Ray.")

    processor: Optional["ProcessorConfig"]  # noqa: F821

    trainer: Optional["TrainerConfig"]  # noqa: F821

    loader: Optional["LoaderConfig"]  # noqa: F821
