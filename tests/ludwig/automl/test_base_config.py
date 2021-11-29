import random

import pytest

from ludwig.automl.base_config import infer_type, should_exclude
from ludwig.automl.utils import FieldInfo
from ludwig.constants import BINARY, CATEGORY, IMAGE, NUMERICAL, TEXT
from ludwig.data.dataset_synthesizer import generate_string

ROW_COUNT = 100
TARGET_NAME = "target"


@pytest.mark.parametrize(
    "num_distinct_values,distinct_values,img_values,missing_vals,expected",
    [
        # Random numbers.
        (ROW_COUNT, [str(random.random()) for _ in range(ROW_COUNT)], 0, 0.0, NUMERICAL),
        # Random numbers with NaNs.
        (ROW_COUNT, [str(random.random()) for _ in range(ROW_COUNT - 1)] + ["NaN"], 0, 0.0, NUMERICAL),
        # Finite list of numbers.
        (10, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 0, 0.0, CATEGORY),
        (2, ["1.5", "3.7"], 0, 0.1, CATEGORY),
        # Bool-like values.
        (2, ["0", "1"], 0, 0.0, BINARY),
        # Mostly bool-like values.
        (3, ["0", "1", "True"], 0, 0.0, CATEGORY),
        # Finite list of strings.
        (2, ["human", "bot"], 0, 0.0, CATEGORY),
        (10, [generate_string(5) for _ in range(10)], 0, 0.0, CATEGORY),
        # Random strings.
        (ROW_COUNT, [generate_string(5) for _ in range(ROW_COUNT)], 0, 0.0, TEXT),
        # Images.
        (ROW_COUNT, [], ROW_COUNT, 0.0, IMAGE),
    ],
)
def test_infer_type(num_distinct_values, distinct_values, img_values, missing_vals, expected):
    field = FieldInfo(
        name="foo",
        dtype="object",
        num_distinct_values=num_distinct_values,
        distinct_values=distinct_values,
        image_values=img_values,
    )
    assert infer_type(field, missing_vals) == expected


@pytest.mark.parametrize(
    "idx,num_distinct_values,dtype,name,expected",
    [
        (3, ROW_COUNT, NUMERICAL, "id", True),
        (0, ROW_COUNT, NUMERICAL, "foo", True),
        (3, ROW_COUNT, TEXT, "uuid", True),
        (0, ROW_COUNT, TEXT, "name", False),
        (0, ROW_COUNT, NUMERICAL, TARGET_NAME, False),
        (0, ROW_COUNT - 1, NUMERICAL, "id", False),
        (0, 0, CATEGORY, "empty_col", True),
    ],
)
def test_should_exclude(idx, num_distinct_values, dtype, name, expected):
    field = FieldInfo(name=name, dtype=dtype, num_distinct_values=num_distinct_values)
    assert should_exclude(idx, field, dtype, ROW_COUNT, TARGET_NAME) == expected
