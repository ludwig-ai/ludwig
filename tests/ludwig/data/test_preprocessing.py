from ludwig.data.preprocessing import is_input_feature
from tests.integration_tests.utils import text_feature


def test_is_input_feature():
    # Adds encoder when output_feature=False
    assert is_input_feature(text_feature(output_feature=False)) is True
    # Adds decoder when output_feature=True
    assert is_input_feature(text_feature(output_feature=True)) is False
