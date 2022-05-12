import random

import pytest

from ludwig.automl.base_config import infer_type, should_exclude
from ludwig.automl.utils import FieldInfo
from ludwig.constants import AUDIO, BINARY, CATEGORY, DATE, IMAGE, NUMBER, TEXT
from ludwig.data.dataset_synthesizer import generate_string

ROW_COUNT = 100
TARGET_NAME = "target"


@pytest.mark.parametrize(
    "num_distinct_values,distinct_values,img_values,audio_values,missing_vals,expected",
    [
        # Random numbers.
        (ROW_COUNT, [str(random.random()) for _ in range(ROW_COUNT)], 0, 0, 0.0, NUMBER),
        # Random numbers with NaNs.
        (ROW_COUNT, [str(random.random()) for _ in range(ROW_COUNT - 1)] + ["NaN"], 0, 0, 0.0, NUMBER),
        # Finite list of numbers.
        (10, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 0, 0, 0.0, CATEGORY),
        (2, ["1.5", "3.7"], 0, 0, 0.1, NUMBER),
        (2, ["1.5", "3.7", "nan"], 0, 0, 0.1, NUMBER),
        # Bool-like values.
        (2, ["0", "1"], 0, 0, 0.0, BINARY),
        # Mostly bool-like values.
        (3, ["0", "1", "True"], 0, 0, 0.0, CATEGORY),
        # Finite list of strings.
        (2, ["human", "bot"], 0, 0, 0.0, CATEGORY),
        (10, [generate_string(5) for _ in range(10)], 0, 0, 0.0, CATEGORY),
        (40, [generate_string(5) for _ in range(40)], 0, 0, 0.0, CATEGORY),
        # Mostly random strings.
        (90, [generate_string(5) for _ in range(90)], 0, 0, 0.0, TEXT),
        # Mostly random strings with capped distinct values.
        (90, [generate_string(5) for _ in range(10)], 0, 0, 0.0, TEXT),
        # All random strings.
        (ROW_COUNT, [generate_string(5) for _ in range(ROW_COUNT)], 0, 0, 0.0, TEXT),
        # Images.
        (ROW_COUNT, [], ROW_COUNT, 0, 0.0, IMAGE),
        # Audio.
        (ROW_COUNT, [], 0, ROW_COUNT, 0.0, AUDIO),
    ],
)
def test_infer_type(num_distinct_values, distinct_values, img_values, audio_values, missing_vals, expected):
    field = FieldInfo(
        name="foo",
        dtype="object",
        num_distinct_values=num_distinct_values,
        distinct_values=distinct_values,
        image_values=img_values,
        audio_values=audio_values,
    )
    assert infer_type(field, missing_vals, ROW_COUNT) == expected


def test_infer_type_explicit_date():
    field = FieldInfo(
        name="foo",
        distinct_values=["1", "2"],
        num_distinct_values=2,
        dtype=DATE,
    )
    assert infer_type(field, 0, ROW_COUNT) == DATE


@pytest.mark.parametrize(
    "idx,num_distinct_values,dtype,name,expected",
    [
        (3, ROW_COUNT, NUMBER, "id", True),
        (0, ROW_COUNT, NUMBER, "foo", True),
        (3, ROW_COUNT, TEXT, "uuid", True),
        (0, ROW_COUNT, TEXT, "name", False),
        (0, ROW_COUNT, NUMBER, TARGET_NAME, False),
        (0, ROW_COUNT - 1, NUMBER, "id", False),
        (0, 0, CATEGORY, "empty_col", True),
    ],
)
def test_should_exclude(idx, num_distinct_values, dtype, name, expected):
    field = FieldInfo(name=name, dtype=dtype, num_distinct_values=num_distinct_values)
    assert should_exclude(idx, field, dtype, ROW_COUNT, TARGET_NAME) == expected
