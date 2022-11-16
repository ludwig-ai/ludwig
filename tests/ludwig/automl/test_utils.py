import numpy as np
import pandas as pd
import pytest

ray = pytest.importorskip("ray")  # noqa

from ludwig.automl.base_config import get_dataset_info  # noqa
from ludwig.utils.automl.utils import get_model_type  # noqa

pytestmark = pytest.mark.distributed


def _features(*in_types, out):
    return {
        "input_features": [{"name": f"in_{i}", "type": dtype} for i, dtype in enumerate(in_types)],
        "output_features": [{"name": "out_0", "type": out}],
    }


@pytest.mark.parametrize(
    "config,expected",
    [
        ({**_features("text", out="number")}, "text"),
        ({**_features("text", "text", out="number")}, "concat"),
        ({**_features("text", "text", out="number"), "combiner": {"type": "tabnet"}}, "tabnet"),
    ],
)
def test_get_model_type(config, expected):
    actual = get_model_type(config)
    assert actual == expected


@pytest.mark.parametrize("missing_value", [(None), (np.nan)], ids=["none", "np.nan"])
def test_bool_type_inference_from_object_dtype(missing_value):
    df = pd.DataFrame({"col1": ["a", "b", "a", "a", "b", missing_value]})
    info = get_dataset_info(df)
    assert info.fields[0].dtype == "bool"


def test_object_type_inference_from_object_dtype():
    df = pd.DataFrame({"col1": ["a", "b", "c", "d", "e", "a", "b", "b"]})
    info = get_dataset_info(df)
    assert info.fields[0].dtype == "object"
