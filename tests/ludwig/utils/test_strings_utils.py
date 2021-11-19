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


def test_are_conventional_bools():
    assert strings_utils.are_conventional_bools(['True', 'False']) == True
    assert strings_utils.are_conventional_bools(['T', 'F']) == True
    assert strings_utils.are_conventional_bools(['t', 'f']) == True
    assert strings_utils.are_conventional_bools(['True', 'Fales']) == False
    assert strings_utils.are_conventional_bools(['0', '1']) == True
    assert strings_utils.are_conventional_bools(['0', '2']) == False
    assert strings_utils.are_conventional_bools(['high', 'low']) == False
    assert strings_utils.are_conventional_bools(['human', 'bot']) == False
