from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json, LetterCase

from ludwig.api_annotations import DeveloperAPI


@DeveloperAPI
@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldInfo:
    name: str
    dtype: str
    key: str = None
    distinct_values: List = None
    distinct_values_balance: float = 1.0
    num_distinct_values: int = 0
    nonnull_values: int = 0
    image_values: int = 0
    audio_values: int = 0
    avg_words: int = None


@DeveloperAPI
@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldConfig:
    name: str
    column: str
    type: str


@DeveloperAPI
@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldMetadata:
    name: str
    config: FieldConfig
    excluded: bool
    mode: str
    missing_values: float
    imbalance_ratio: float
