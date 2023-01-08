from abc import ABC

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

model_type_schema_registry = Registry()


@DeveloperAPI
@dataclass(repr=False)
class BaseModelTypeConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for optimizers. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding `torch.optim.Optimizer` class are copied over: check each class to check which attributes are
    different from the torch-specified defaults.
    """

    model_type: str


@DeveloperAPI
def register_model_type(name: str):
    def wrap(model_type_config: BaseModelTypeConfig) -> BaseModelTypeConfig:
        model_type_schema_registry[name] = model_type_config
        return model_type_config

    return wrap
