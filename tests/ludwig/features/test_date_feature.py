from copy import deepcopy
from datetime import date, datetime
from typing import Any, List

import pytest
import torch
from dateutil.parser import parse

from ludwig.constants import ENCODER_OUTPUT, FILL_WITH_CONST, MISSING_VALUE_STRATEGY
from ludwig.features import date_feature
from ludwig.features.date_feature import DateInputFeature
from ludwig.schema.features.date_feature import DateInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.types import FeatureConfigDict
from ludwig.utils.date_utils import create_vector_from_datetime_obj
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DATE_W_SIZE = 9
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def date_config():
    return {"name": "date_column_name", "type": "date"}


def test_date_input_feature(date_config: FeatureConfigDict):
    # setup image input feature definition
    feature_def = deepcopy(date_config)

    # pickup any other missing parameters
    defaults = DateInputFeatureConfig(name="foo").to_dict()
    set_def = merge_dict(defaults, feature_def)

    # ensure no exceptions raised during build
    feature_config, _ = load_config_with_kwargs(DateInputFeatureConfig, set_def)
    input_feature_obj = DateInputFeature(feature_config).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = input_feature_obj.create_sample_input(batch_size=BATCH_SIZE)
    assert input_tensor.shape == torch.Size((BATCH_SIZE, DATE_W_SIZE))
    assert input_tensor.dtype == torch.int32

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output[ENCODER_OUTPUT].shape == (BATCH_SIZE, *input_feature_obj.output_shape)


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


@pytest.fixture(scope="module")
def reference_date_list() -> List[int]:
    return create_vector_from_datetime_obj(datetime.utcfromtimestamp(1691600953.443032))


@pytest.fixture(scope="module")
def fill_value() -> str:
    return "1970-01-01 00:00:00"


@pytest.fixture(scope="module")
def fill_value_list(fill_value: str) -> List[int]:
    return create_vector_from_datetime_obj(parse(fill_value))


@pytest.mark.parametrize(
    "timestamp,datetime_format,expected_list",
    [
        pytest.param(1691600953.443032, None, "reference_date_list", id="float-s"),
        pytest.param(1691600953443.032, None, "reference_date_list", id="float-ms"),
        pytest.param(1691600953, None, "reference_date_list", id="int-s"),
        pytest.param(1691600953443, None, "reference_date_list", id="int-ms"),
        pytest.param(1691600953.443032, "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="float-s-fmt"),
        pytest.param(1691600953443.032, "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="float-ms-fmt"),
        pytest.param(1691600953, "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="int-s-fmt"),
        pytest.param(1691600953443, "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="int-ms-fmt"),
        pytest.param("1691600953.443032", None, "reference_date_list", id="string[float]-s"),
        pytest.param("1691600953443.0032", None, "reference_date_list", id="string[float]-ms"),
        pytest.param("1691600953", None, "reference_date_list", id="string[int]-s"),
        pytest.param("1691600953443", None, "reference_date_list", id="string[int]-ms"),
        pytest.param("1691600953.443032", "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="string[float]-s-fmt"),
        pytest.param("1691600953443.0032", "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="string[float]-ms-fmt"),
        pytest.param("1691600953", "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="string[int]-s-fmt"),
        pytest.param("1691600953443", "%d/%m/%y %H:%M:%S.%f", "reference_date_list", id="string[int]-ms-fmt"),
        pytest.param("foo", None, "fill_value_list", id="string error"),
        pytest.param([1691600953.443032], None, "fill_value_list", id="list error"),
        pytest.param(None, None, "fill_value_list", id="NoneType error"),
    ],
)
def test_date_to_list_numeric(timestamp: Any, datetime_format: str, expected_list: List[int], fill_value: str, request):
    """Test that numeric datetime formats are converted correctly.

    Currently, we support int, float, and string representations of POSIX timestamps in seconds and milliseconds. Valid
    timestamps should be converted to datetime lists by `luwdig.utils.date_utils.create_vector_from_datetime_object`.
    If a string format is provided, it should be ignored.

    Args:
        timestamp: Input to be converted to a date vector
        datetime_format: Optional format string, should be ignored under the hood with these timestamps.
        expected_list: The expected output of `DateFeatureMixin.date_to_list`
        fill_value: Date to be used as fallback
        request: pytest request fixture
    """
    expected_result = request.getfixturevalue(expected_list)

    # The default fill value is `datetime.now`, for testing we override this to be a constant.
    preprocessing_parameters = {MISSING_VALUE_STRATEGY: FILL_WITH_CONST, "fill_value": fill_value}

    # No exception should ever be raised from `date_to_list` due to a parsing error. The expected behavior is to fall
    # back to the fill value.
    dt = date_feature.DateInputFeature.date_to_list(timestamp, datetime_format, preprocessing_parameters)
    assert dt == expected_result


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


@pytest.fixture(scope="module")
def date_obj():
    return date.fromisoformat("2022-06-25")


@pytest.fixture(scope="module")
def date_obj_vec():
    return create_vector_from_datetime_obj(datetime.fromisoformat("2022-06-25"))


def test_date_object_to_list(date_obj, date_obj_vec, fill_value):
    """Test support for datetime.date object conversion.

    Args:
        date_obj: Date object to convert into a vector
        date_obj_vector: Expected vector version of `date_obj`
    """
    computed_date_vec = date_feature.DateInputFeature.date_to_list(
        date_obj, None, preprocessing_parameters={MISSING_VALUE_STRATEGY: FILL_WITH_CONST, "fill_value": fill_value}
    )
    assert computed_date_vec == date_obj_vec
