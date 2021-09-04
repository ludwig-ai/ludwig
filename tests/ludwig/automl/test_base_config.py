import pytest

from ludwig.automl.base_config import should_exclude, infer_type
from ludwig.automl.utils import FieldInfo
from ludwig.constants import TEXT, NUMERICAL, CATEGORY, BINARY, IMAGE

ROW_COUNT = 100
TARGET_NAME = 'target'


@pytest.mark.parametrize("distinct_values,avg_words,img_values,expected", [
    (ROW_COUNT, 0, 0, NUMERICAL),
    (10, 0, 0, CATEGORY),
    (2, 0, 0, BINARY),
    (ROW_COUNT, 3, 0, TEXT),
    (ROW_COUNT, 1, ROW_COUNT, IMAGE),
])
def test_infer_type(distinct_values, avg_words, img_values, expected):
    field = FieldInfo(
        name='foo',
        dtype='object',
        distinct_values=distinct_values,
        avg_words=avg_words,
        image_values=img_values,
    )
    assert infer_type(field) == expected


@pytest.mark.parametrize("idx,distinct_values,dtype,name,expected", [
    (3, ROW_COUNT, NUMERICAL, 'id', True),
    (0, ROW_COUNT, NUMERICAL, 'foo', True),
    (3, ROW_COUNT, TEXT, 'uuid', True),
    (0, ROW_COUNT, TEXT, 'name', False),
    (0, ROW_COUNT, NUMERICAL, TARGET_NAME, False),
    (0, ROW_COUNT - 1, NUMERICAL, 'id', False),
])
def test_should_exclude(idx, distinct_values, dtype, name, expected):
    field = FieldInfo(
        name=name,
        dtype=dtype,
        distinct_values=distinct_values
    )
    assert should_exclude(idx, field, dtype, ROW_COUNT, TARGET_NAME) == expected
