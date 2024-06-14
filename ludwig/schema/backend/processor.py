from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.backend.utils import register_processor_config
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseProcessorConfig(schema_utils.BaseMarshmallowConfig):
    """Configuration for distributed data processing (only supported by the `ray` backend)."""

    type: str = schema_utils.StringOptions(
        options=["dask", "modin"],
        default="dask",
        description='Distributed data processing engine to use. `"dask"`: (default) a lazily executed version of '
        'distributed Pandas. `"modin"`: an eagerly executed version of distributed Pandas.',
    )


@DeveloperAPI
@register_processor_config("dask")
@ludwig_dataclass
class DaskProcessorConfig(BaseProcessorConfig):
    type: str = schema_utils.ProtectedString("dask")

    parallelism: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="(dask only) The number of partitions to divide the dataset into (defaults to letting Dask figure "
        "this out automatically).",
    )

    persist: bool = schema_utils.Boolean(
        default=True,
        description="(dask only) Whether intermediate stages of preprocessing should be cached in distributed memory.",
    )


@DeveloperAPI
@register_processor_config("modin")
@ludwig_dataclass
class ModinProcessorConfig(BaseProcessorConfig):
    type: str = schema_utils.ProtectedString("modin")


@DeveloperAPI
def ProcessorDataclassField(description: str = "", default: Dict = {}):
    class ProcessorMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return BaseProcessorConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for processor: {value}, see BaseProcessorConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                "type": "object",
                "title": "processor_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: BaseProcessorConfig.Schema().load(default)
    dump_default = BaseProcessorConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": ProcessorMarshmallowField(
                allow_none=False,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=load_default,
    )
