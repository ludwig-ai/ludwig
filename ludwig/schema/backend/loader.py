from dataclasses import field
from typing import Dict, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseLoaderConfig(schema_utils.BaseMarshmallowConfig):
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
        field_options=[
            schema_utils.PositiveInteger("Dataset window size in bytes.", default=None),
            schema_utils.ProtectedString("auto"),
        ],
        allow_none=True,
        description="Load and shuffle the preprocessed dataset in discrete windows of this size (defaults to null, "
        "meaning data will not be windowed). Try configuring this is if shuffling is taking a very long "
        "time, indicated by every epoch of training taking many minutes to start. In general, larger "
        "window sizes result in more uniform shuffling (which can lead to better model performance in "
        "some cases), while smaller window sizes will be faster to load. This setting is particularly "
        "useful when running hyperopt over a large dataset.",
    )


@DeveloperAPI
def LoaderDataclassField(description: str = "", default: Dict = {}):
    class LoaderMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return BaseLoaderConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for loader: {value}, see LoaderConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                "type": "object",
                "title": "loader_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: BaseLoaderConfig.Schema().load(default)
    dump_default = BaseLoaderConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": LoaderMarshmallowField(
                allow_none=False,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=load_default,
    )
