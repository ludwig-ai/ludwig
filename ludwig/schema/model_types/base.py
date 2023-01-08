from abc import ABC
from typing import Any, Dict, List, Optional
from marshmallow import ValidationError

from marshmallow_dataclass import dataclass
import marshmallow_dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.utils.registry import Registry

model_type_schema_registry = Registry()


@DeveloperAPI
@dataclass(repr=False)
class BaseModelTypeConfig(schema_utils.BaseMarshmallowConfig, ABC):
    input_features: List[BaseInputFeatureConfig]
    output_features: List[BaseOutputFeatureConfig]

    model_type: str

    trainer: BaseTrainerConfig
    preprocessing: PreprocessingConfig
    defaults: DefaultsConfig
    hyperopt: Optional[HyperoptConfig] = None

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "BaseModelTypeConfig":
        model_type = config.get("model_type", MODEL_ECD)
        if model_type not in model_type_schema_registry:
            raise ValidationError(
                f"Invalid model type: '{model_type}', expected one of: {list(model_type_schema_registry.keys())}"
            )
        cls = model_type_schema_registry[model_type]
        schema = marshmallow_dataclass.class_schema(cls)()
        return schema.load(config)


@DeveloperAPI
def register_model_type(name: str):
    def wrap(model_type_config: BaseModelTypeConfig) -> BaseModelTypeConfig:
        model_type_schema_registry[name] = model_type_config
        return model_type_config

    return wrap
