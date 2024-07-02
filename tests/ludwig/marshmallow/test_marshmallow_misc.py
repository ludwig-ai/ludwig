import pytest
from marshmallow_dataclass import dataclass

import ludwig.combiners.combiners as lcc
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.schema.utils import assert_is_a_marshmallow_class, BaseMarshmallowConfig, load_config_with_kwargs


@dataclass
class CustomTestSchema(BaseMarshmallowConfig):
    """Sample docstring."""

    foo: int = 5
    "foo (default: 5)"


def test_assert_is_a_marshmallow_clas():
    assert_is_a_marshmallow_class(ECDTrainerConfig)
    with pytest.raises(AssertionError, match=r"^Expected marshmallow class.*"):
        assert_is_a_marshmallow_class(lcc.ConcatCombiner)


def test_load_config_with_kwargs():
    test_kwargs = {
        "foo": 6,
        "bar": 6,
    }
    initialized_class, leftover = load_config_with_kwargs(CustomTestSchema, test_kwargs)

    assert initialized_class.foo == 6
    assert leftover == {"bar": 6}

    # TransformerCombiner has no required/non-default arguments:
    initialized_class, leftover = load_config_with_kwargs(lcc.TransformerCombinerConfig, test_kwargs)
    assert initialized_class.bias_initializer == "zeros"
    assert leftover == test_kwargs
    initialized_class, leftover = load_config_with_kwargs(lcc.TransformerCombinerConfig, {})
    assert leftover == {}
