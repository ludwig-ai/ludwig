import pandas as pd
import pytest

try:
    from ludwig.automl.utils import avg_num_tokens
except ImportError:
    pass


@pytest.mark.parametrize("field,expected", [(pd.Series([None]), 0), (pd.Series(["string1", "string2", "string3"]), 1)])
@pytest.mark.distributed
def test_avg_num_tokens(field, expected):
    assert avg_num_tokens(field) == expected
