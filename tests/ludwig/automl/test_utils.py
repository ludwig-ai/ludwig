import numpy as np
import pytest

from ludwig.automl.utils import avg_num_tokens

np.asarray([None])


@pytest.mark.parametrize(
    "field,expected",
    [
        (np.asarray([None]), 0),
        (np.asarray(["string1", "string2", "string3"]), 1)
    ]
)
def test_avg_num_tokens(field, expected):
    assert avg_num_tokens(field) == expected
