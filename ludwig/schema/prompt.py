from ludwig.api_annotations import DeveloperAPI

from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class RetrievalConfig(schema_utils.DictMarshmallowField):
    """This Dataclass is a schema for the nested retrieval config under prompt."""
    
    type: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The type of retrieval to use for the prompt.",
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
@ludwig_dataclass
class PromptConfig(schema_utils.DictMarshmallowField):
    """This Dataclass is a schema for the nested prompt config under preprocessing."""
    
    retrieval: RetrievalConfig
    
    task: str = schema_utils.String(
        default="Given the sample input. Complete this sentence by replacing XXXX: The label is XXXX.",
        allow_none=True,
        description="The task to use for the prompt.",
    )