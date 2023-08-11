import datetime
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager

import pytest

from ludwig.utils.date_utils import convert_number_to_datetime


@pytest.fixture(scope="module")
def reference_datetime() -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(1691600953.443032)


@pytest.mark.parametrize(
    "timestamp,raises",
    [
        pytest.param(1691600953.443032, does_not_raise(), id="float-s"),
        pytest.param(1691600953443.032, does_not_raise(), id="float-ms"),
        pytest.param(1691600953, does_not_raise(), id="int-s"),
        pytest.param(1691600953443, does_not_raise(), id="int-ms"),
        pytest.param("1691600953.443032", does_not_raise(), id="string[float]-s"),
        pytest.param("1691600953443.0032", does_not_raise(), id="string[float]-ms"),
        pytest.param("1691600953", does_not_raise(), id="string[int]-s"),
        pytest.param("1691600953443", does_not_raise(), id="string[int]-ms"),
        pytest.param("foo", pytest.raises(ValueError), id="string error"),
        pytest.param([1691600953.443032], pytest.raises(ValueError), id="list error"),
        pytest.param(datetime.datetime(2023, 8, 9, 13, 9, 13), pytest.raises(ValueError), id="datetime error"),
        pytest.param(None, pytest.raises(ValueError), id="NoneType error"),
    ],
)
def test_convert_number_to_datetime(reference_datetime: datetime.datetime, timestamp: Any, raises: ContextManager):
    """Ensure that numeric timestamps are correctly converted to datetime objects.

    Args:
        reference_datetime: A datetime object with the expected date/time
        timestamp: The timestamp to convert in s or ms
        raises: context manager to check for expected exceptions
    """
    with raises:
        dt = convert_number_to_datetime(timestamp)

        # Check that the returned datetime is accurate to the scale of seconds.
        assert dt.year == reference_datetime.year
        assert dt.month == reference_datetime.month
        assert dt.day == reference_datetime.day
        assert dt.hour == reference_datetime.hour
        assert dt.minute == reference_datetime.minute
        assert dt.second == reference_datetime.second
