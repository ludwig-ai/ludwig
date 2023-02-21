from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BackendConfig(schema_utils.BaseMarshmallowConfig):
    """Global backend compute resource/usage configuration."""

    type: str = schema_utils.StringOptions(
        ["local", "ray", "ray"],
        "local",
        description='How the job will be distributed, one of "local", "ray", or "horovod".',
    )
