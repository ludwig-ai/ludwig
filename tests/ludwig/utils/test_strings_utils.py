import pytest

from ludwig.utils import strings_utils


def test_str_to_bool():
    # Global bool mappings are used.
    assert strings_utils.str2bool('True') == True
    assert strings_utils.str2bool('true') == True
    assert strings_utils.str2bool('0') == False

    # Error raised if non-mapped value is encountered and no fallback is specified.
    with pytest.raises(Exception):
        strings_utils.str2bool('bot')

    # Fallback label is used.
    assert strings_utils.str2bool('bot', fallback_true_label='bot') == True
    assert strings_utils.str2bool('human', fallback_true_label='bot') == False
    assert strings_utils.str2bool('human', fallback_true_label='human') == True
    assert strings_utils.str2bool(
        'human', fallback_true_label='Human') == False

    # Fallback label is used, strictly as a fallback.
    assert strings_utils.str2bool('True', fallback_true_label='False') == True
