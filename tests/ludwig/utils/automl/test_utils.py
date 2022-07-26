import pandas as pd
import pytest

from ludwig.utils.automl.utils import avg_num_tokens


@pytest.mark.parametrize("field,expected", [(pd.Series([None]), 0), (pd.Series(["string1", "string2", "string3"]), 1)])
def test_avg_num_tokens(field, expected):
    assert avg_num_tokens(field) == expected
