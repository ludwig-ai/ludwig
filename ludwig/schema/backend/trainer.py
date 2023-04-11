from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


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
