from copy import deepcopy
from datetime import datetime

import pytest
import torch

from ludwig.features import date_feature
from ludwig.features.date_feature import DateInputFeature
from ludwig.schema.features.date_feature import DateInputFeatureConfig
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.types import FeatureConfigDict
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
    defaults = DateInputFeatureConfig().to_dict()
    set_def = merge_dict(defaults, feature_def)

    # ensure no exceptions raised during build
    feature_config, _ = load_config_with_kwargs(DateInputFeatureConfig, set_def)
    input_feature_obj = DateInputFeature(feature_config).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = input_feature_obj.create_sample_input(batch_size=BATCH_SIZE)
    assert input_tensor.shape == torch.Size((BATCH_SIZE, DATE_W_SIZE))
    assert input_tensor.dtype == torch.int32

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)


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
