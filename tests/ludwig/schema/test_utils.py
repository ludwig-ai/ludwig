from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils


def test_remove_duplicate_fields():
    props = {TYPE: "random", "probabilities": [0.7, 0.1, 0.2]}
    schema_utils.remove_duplicate_fields(props, [TYPE])
    assert TYPE not in props
    assert "probabilities" in props
