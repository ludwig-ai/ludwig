import pytest

ray = pytest.importorskip("ray")  # noqa

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
