from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.backend.loader import BaseLoaderConfig, LoaderDataclassField
from ludwig.schema.backend.processor import BaseProcessorConfig, ProcessorDataclassField
from ludwig.schema.backend.trainer import BackendTrainerDataclassField, BaseBackendTrainerConfig
from ludwig.schema.backend.utils import register_backend_config
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_backend_config("local")
@register_backend_config("horovod")
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
@register_backend_config("ray")
@ludwig_dataclass
class RayBackendConfig(BaseBackendConfig):
    type: str = schema_utils.ProtectedString("ray", description="Distribute training with Ray.")

    processor: BaseProcessorConfig = ProcessorDataclassField()

    trainer: BaseBackendTrainerConfig = BackendTrainerDataclassField()

    loader: BaseLoaderConfig = LoaderDataclassField()


@DeveloperAPI
def get_backend_jsonschema():
    props = schema_utils.unload_jsonschema_from_marshmallow_class(BaseBackendConfig)["properties"]

    return {
        "type": ["object", "null"],
        "properties": props,
        "title": "backend_options",
        "description": "Settings for computational backend",
    }


@DeveloperAPI
class BackendField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(BaseBackendConfig, default_missing=True)

    def _jsonschema_type_mapping(self):
        return get_backend_jsonschema()
