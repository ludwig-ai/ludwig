from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SEMANTIC
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class RetrievalConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a schema for the nested retrieval config under prompt."""

    def __post_init__(self):
        # TODO: have a dynamically loaded schema based on the selection of the type param
        # https://github.com/ludwig-ai/ludwig/pull/3351#discussion_r1181910954
        # Ensure k is non-zero if we're using a retrieval strategy
        if self.type is not None and self.k == 0:
            self.k = 1

        if self.type is None and self.k != 0:
            raise ConfigValidationError("k must be 0 if retrieval type is None.")
        elif self.type is not None and self.k <= 0:
            raise ConfigValidationError("k must be greater than 0 if retrieval type is not None.")

        if self.type is None and self.model_name is not None:
            raise ConfigValidationError("model_name must be None if retrieval type is None.")
        elif self.type == SEMANTIC and self.model_name is None:
            raise ConfigValidationError(f"model_name must not be None if retrieval type is '{SEMANTIC}'.")

    type: str = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "The type of retrieval to use for the prompt. If `None`, then no retrieval is used, and the task "
            "is framed as a zero-shot learning problem. If not `None` (e.g. either 'random' or 'semantic'), then "
            "samples are retrieved from an index of the training set and used to augment the input to the model "
            "in a few-shot learning setting."
        ),
        parameter_metadata=LLM_METADATA["prompt"]["retrieval"]["type"],
    )

    index_name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The name of the index to use for the prompt. Indices are stored in the ludwig cache by default.",
        parameter_metadata=LLM_METADATA["prompt"]["retrieval"]["index_name"],
    )

    model_name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The model used to generate the embeddings used to retrieve samples to inject in the prompt.",
        parameter_metadata=LLM_METADATA["prompt"]["retrieval"]["model_name"],
    )

    k: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of samples to retrieve.",
        parameter_metadata=LLM_METADATA["prompt"]["retrieval"]["k"],
    )


@DeveloperAPI
class RetrievalConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(RetrievalConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(RetrievalConfig, title="Retrieval")


@DeveloperAPI
@ludwig_dataclass
class PromptConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a schema for the nested prompt config under preprocessing."""

    template: str = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "The template to use for the prompt. Must contain at least one of the columns from the input dataset "
            "or `__sample__` as a variable surrounded in curly brackets {} to indicate where to insert the "
            "current feature. Multiple columns can be inserted, e.g.: `The {color} {animal} jumped over "
            "the {size} {object}`, where every term in curly brackets is a column in the dataset. If a `task` "
            "is specified, then the template must also contain the `__task__` variable. If `retrieval` is specified, "
            "then the template must also contain the `__context__` variable. If no template is provided, then a "
            "default will be used based on the retrieval settings, and a task must be set in the config."
        ),
        parameter_metadata=LLM_METADATA["prompt"]["template"],
    )

    task: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The task to use for the prompt. Required if `template` is not set.",
        parameter_metadata=LLM_METADATA["prompt"]["task"],
    )

    retrieval: RetrievalConfig = RetrievalConfigField().get_default_field()


@DeveloperAPI
class PromptConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(PromptConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(PromptConfig)
