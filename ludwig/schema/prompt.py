from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class RetrievalConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a schema for the nested retrieval config under prompt."""

    type: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The type of retrieval to use for the prompt.",
    )

    index_name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The name of the index to use for the prompt. Indices are stored in the ludwig cache by default.",
    )

    model_name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The model to use for the prompt.",
    )

    k: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of samples to retrieve.",
    )


@DeveloperAPI
class RetrievalConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(RetrievalConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(RetrievalConfig)


@DeveloperAPI
@ludwig_dataclass
class PromptConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a schema for the nested prompt config under preprocessing."""

    retrieval: RetrievalConfig = RetrievalConfigField().get_default_field()

    task: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The task to use for the prompt.",
    )

    template: str = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "Advanced: the template to use for the prompt. Must contain `context`, `sample_input` and `task` "
            "variables. `context` is the placeholder for labeled samples, `sample_input` is the placeholder for a "
            "single, unlabeled sample, and `task` is the placeholder for the user-specified task description."
        ),
    )


@DeveloperAPI
class PromptConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(PromptConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(PromptConfig)
