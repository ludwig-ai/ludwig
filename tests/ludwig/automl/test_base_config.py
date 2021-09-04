import pytest

from ludwig.automl.base_config import should_exclude
from ludwig.automl.utils import FieldInfo
from ludwig.constants import TEXT, NUMERICAL


ROW_COUNT = 100
TARGET_NAME = 'target'


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
