import numpy as np
import pytest

from ludwig.utils.numerical_test_utils import assert_all_finite


@pytest.fixture
def finite_valued_dict():
    return {
        "scalar": 1,
        "metrics": {"val": 0.2, "series": [0.1, 0.2, 0.3], "ndarray": np.ones((8, 4, 2))},
    }


def test_assert_all_finite(finite_valued_dict):
    assert_all_finite(finite_valued_dict)


def test_fail_with_nan(finite_valued_dict):
    finite_valued_dict["scalar"] = float("nan")
    with pytest.raises(Exception):
        assert_all_finite(finite_valued_dict)


def test_fail_with_inf(finite_valued_dict):
    finite_valued_dict["scalar"] = float("inf")
    with pytest.raises(Exception):
        assert_all_finite(finite_valued_dict)


def test_fail_with_nan_in_list(finite_valued_dict):
    finite_valued_dict["scalar"] = float("nan")
    with pytest.raises(Exception):
        assert_all_finite(finite_valued_dict)


def test_fail_with_nan_in_ndarray(finite_valued_dict):
    finite_valued_dict["metrics"]["ndarray"][0, 0, 1] = np.nan
    with pytest.raises(Exception):
        assert_all_finite(finite_valued_dict)
