# from abc import ABC
# from typing import List, Optional, Tuple, Union

# from ludwig.api_annotations import DeveloperAPI
# from ludwig.schema import utils as schema_utils
# from ludwig.utils.registry import Registry


# llm_adapter_registry = Registry()


# @DeveloperAPI
# def register_adapter(name: str):
#     def wrap(adapter_config: LLMBaseAdapterConfig):
#         llm_adapter_registry[name] = adapter_config
#         return adapter_config

#     return wrap


# @DeveloperAPI
# def get_adapter_jsonschema(adapter_type: str):
#     llm_adapter = llm_adapter_registry[adapter_type]
#     props = schema_utils.unload_jsonschema_from_marshmallow_class(llm_adapter)["properties"]

#     return {
#         "type": ["object", "null"],
#         "properties": props,
#         "title": "adapter_options",
#         "description": "Schema for fine tuning adapters for LLMs",
#     }


# @DeveloperAPI
# @schema_utils.ludwig_dataclass
# class LLMBaseAdapterConfig(schema_utils.BaseMarshmallowConfig, ABC):
#     pass


# @DeveloperAPI
# @schema_utils.ludwig_dataclass
# @register_adapter("prompt_tuning")
# class LLMPromptTuningAdapterConfig(LLMBaseAdapterConfig):
#     pass


# @DeveloperAPI
# @schema_utils.ludwig_dataclass
# @DeveloperAPI
# class LLMAdapterConfigField(schema_utils.DictMarshmallowField):
#     def __init__(self):
#         super().__init__(LLMBaseAdapterConfig)

#     def _jsonschema_type_mapping(self):
#         return get_adapter_jsonschema()
