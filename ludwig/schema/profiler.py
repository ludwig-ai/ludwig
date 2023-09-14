from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ProfilerConfig(schema_utils.BaseMarshmallowConfig):
    """Dataclass that holds profiling parameters for torch profile scheduler.

    The profiler will skip the first skip_first steps, then wait for wait steps, then do the warmup for the next warmup
    steps, then do the active recording for the next active steps and then repeat the cycle starting with wait steps.
    The optional number of cycles is specified with the repeat parameter, the zero value means that the cycles will
    continue until the profiling is finished.
    """

    wait: int = schema_utils.IntegerRange(
        default=1,
        min=0,
        description="The number of steps to wait profiling.",
    )

    warmup: int = schema_utils.IntegerRange(
        default=1,
        min=0,
        description="The number of steps for profiler warmup after waiting finishes.",
    )

    active: int = schema_utils.IntegerRange(
        default=3,
        min=0,
        description="The number of steps that are actively recorded. Values more than 10 wil dramatically slow down "
        "tensorboard loading.",
    )

    repeat: int = schema_utils.IntegerRange(
        default=5,
        min=0,
        description="The optional number of profiling cycles. Use 0 to profile the entire training run.",
    )

    skip_first: int = schema_utils.IntegerRange(
        default=0,
        min=0,
        max=100,
        description="The number of steps to skip in the beginning of training.",
    )


@DeveloperAPI
def ProfilerDataclassField(description: str, default: Dict = {}):
    """Returns custom dataclass field for `ludwig.modules.profiler.ProfilerConfig`. Allows `None` by default.

    :param description: Description of the torch profiler field
    :param default: dict that specifies clipping param values that will be loaded by its schema class (default: {}).
    """
    allow_none = True

    class ProfilingMarshmallowField(fields.Field):
        """Custom marshmallow field class for the torch profiler.

        Deserializes a dict to a valid instance of `ludwig.modules.optimization_modules.ProfilerConfig` and
        creates a corresponding JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return ProfilerConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for profiling config: {value}, see ProfilerConfig class.")
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(ProfilerConfig),
                "title": "profiler_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    def load_default():
        return ProfilerConfig.Schema().load(default)

    dump_default = ProfilerConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": ProfilingMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                    "parameter_metadata": None,
                },
            )
        },
        default_factory=load_default,
    )
