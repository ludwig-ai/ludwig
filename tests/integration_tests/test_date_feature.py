import datetime
import time

import pandas as pd
import pytest
from dateutil.parser import parse

from ludwig.api import LudwigModel
from ludwig.constants import (
    BACKEND,
    BINARY,
    DATE,
    EPOCHS,
    FILL_WITH_CONST,
    INPUT_FEATURES,
    MISSING_VALUE_STRATEGY,
    NAME,
    OUTPUT_FEATURES,
    PREPROCESSING,
    RAY,
    TRAINER,
    TYPE,
)
from ludwig.utils.date_utils import create_vector_from_datetime_obj

ray = pytest.importorskip("ray")

pytestmark = [
    pytest.mark.distributed,
]


@pytest.fixture(scope="module")
def string_date_df() -> "pd.DataFrame":
    df = pd.DataFrame.from_dict(
        {
            "date_feature": [str(datetime.datetime.now()) for i in range(100)],
            "binary_feature": [i % 2 for i in range(100)],
        }
    )
    return df


@pytest.fixture(scope="module")
def int_date_df() -> "pd.DataFrame":
    df = pd.DataFrame.from_dict(
        {
            "date_feature": [time.time_ns() for i in range(100)],
            "binary_feature": [i % 2 for i in range(100)],
        }
    )
    return df


@pytest.fixture(scope="module")
def float_date_df() -> "pd.DataFrame":
    df = pd.DataFrame.from_dict(
        {
            "date_feature": [time.time() for i in range(100)],
            "binary_feature": [i % 2 for i in range(100)],
        }
    )
    return df


@pytest.mark.parametrize(
    "date_df",
    [
        pytest.param("string_date_df", id="string_date"),
        pytest.param("int_date_df", id="int_date"),
        pytest.param("float_date_df", id="float_date"),
    ],
)
def test_date_feature_formats(date_df, request, ray_cluster_2cpu):
    df = request.getfixturevalue(date_df)

    config = {
        INPUT_FEATURES: [
            {
                NAME: "date_feature",
                TYPE: DATE,
                PREPROCESSING: {MISSING_VALUE_STRATEGY: FILL_WITH_CONST, "fill_value": "1970-01-01 00:00:00"},
            }
        ],
        OUTPUT_FEATURES: [{NAME: "binary_feature", TYPE: BINARY}],
        TRAINER: {EPOCHS: 2},
        BACKEND: {TYPE: RAY, "processor": {TYPE: "dask"}},
    }

    fill_value = create_vector_from_datetime_obj(parse("1970-01-01 00:00:00"))

    model = LudwigModel(config)
    preprocessed = model.preprocess(df)

    # Because parsing errors are suppressed, we want to ensure that the data was preprocessed correctly. Sample data is
    # drawn from the current time, so the recorded years should not match the fill value's year.
    for date in preprocessed.training_set.to_df().compute().iloc[:, 0].values:
        assert date[0] != fill_value[0]

    for date in preprocessed.validation_set.to_df().compute().iloc[:, 0].values:
        assert date[0] != fill_value[0]

    for date in preprocessed.test_set.to_df().compute().iloc[:, 0].values:
        assert date[0] != fill_value[0]
