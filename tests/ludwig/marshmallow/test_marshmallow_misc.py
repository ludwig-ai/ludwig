import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow.utils import EXCLUDE
from marshmallow_dataclass import dataclass

import ludwig.combiners.combiners as lcc
import ludwig.validation.schema as lusutils
from ludwig.models.trainer import TrainerConfig
from ludwig.validation.schema import BaseMarshmallowConfig


@dataclass
class CustomTestSchema(BaseMarshmallowConfig):
    """sample docstring."""

    foo: int = 5
    "foo (default: 5)"


def test_assert_is_a_marshmallow_clas():
    lusutils.assert_is_a_marshmallow_class(TrainerConfig)
    with pytest.raises(AssertionError, match=r"^Expected marshmallow class.*"):
        lusutils.assert_is_a_marshmallow_class(lcc.ConcatCombiner)


def test_custom_marshmallow_inheritance():
    assert CustomTestSchema.Meta.unknown == EXCLUDE


def test_load_config_with_kwargs():
    test_kwargs = {
        "foo": 6,
        "bar": 6,
    }
    initialized_class, leftover = lusutils.load_config_with_kwargs(CustomTestSchema, test_kwargs)

    assert initialized_class.foo == 6
    assert leftover == {"bar": 6}

    # TransformerCombiner has no required/non-default arguments:
    initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.TransformerCombinerConfig, test_kwargs)
    assert initialized_class.bias_initializer == "zeros"
    assert leftover == test_kwargs
    initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.TransformerCombinerConfig, {})
    assert leftover == {}

    # ComparatorCombiner does have required arguments, so expect a failure:
    with pytest.raises(MarshmallowValidationError):
        initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.ComparatorCombinerConfig, test_kwargs)
