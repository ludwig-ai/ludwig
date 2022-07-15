from datetime import datetime

import pytest

from ludwig.features import date_feature


@pytest.mark.parametrize(
    "date_str,datetime_format,expected_list",
    [
        ("2012-02-26T13:51:50.417-07:00", None, [2012, 2, 26, 6, 57, 13, 51, 50, 49910]),
        ("2022-06-25 09:30:59", None, [2022, 6, 25, 5, 176, 9, 30, 59, 34259]),
        ("2022-06-25", None, [2022, 6, 25, 5, 176, 0, 0, 0, 0]),
    ],
)
def test_date_to_list(date_str, datetime_format, expected_list):
    preprocessing_parameters = None
    assert (
        date_feature.DateInputFeature.date_to_list(date_str, datetime_format, preprocessing_parameters) == expected_list
    )


def test_date_to_list__DatetimeObjectFromParsedJSON():
    preprocessing_parameters = None
    datetime_obj = datetime.fromisoformat("2022-06-25")
    assert date_feature.DateInputFeature.date_to_list(datetime_obj, None, preprocessing_parameters) == [
        2022,
        6,
        25,
        5,
        176,
        0,
        0,
        0,
        0,
    ]


def test_date_to_list__UsesFillValueOnInvalidDate():
    preprocessing_parameters = {"fill_value": "2013-02-26"}
    invalid_date_str = "2012abc-02"
    datetime_format = None
    assert date_feature.DateInputFeature.date_to_list(invalid_date_str, datetime_format, preprocessing_parameters) == [
        2013,
        2,
        26,
        1,
        57,
        0,
        0,
        0,
        0,
    ]
