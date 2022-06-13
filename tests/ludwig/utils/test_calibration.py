import numpy as np
import pytest

from ludwig.utils import calibration


@pytest.fixture
def uncalibrated_logits_and_labels():
    """Returns a pair of logits (10x3) and labels (10)."""
    return (
        np.array(
            [
                [-3.596756, 6.728981, 6.3807454],
                [-16.818138, -3.5217745, -1.7786252],
                [-16.060827, 4.7207646, 3.5336719],
                [-4.784969, 5.062503, 3.515455],
                [-4.669478, 7.171067, 6.5137157],
                [-32.596764, -3.5582566, -5.2003713],
                [-4.4035864, 6.3911495, 4.7273974],
                [-4.2035627, 7.846533, 6.0476217],
                [-20.748848, -3.1521742, -4.873552],
                [-4.8901286, 4.726167, 3.208372],
            ]
        ),
        np.array([2, 0, 2, 1, 1, 2, 0, 1, 0, 1]),
    )


EPSILON = 0.1  # maximum relative precision error allowed.


def test_temperature_scaling_binary(uncalibrated_logits_and_labels):
    logits, labels = uncalibrated_logits_and_labels
    # Selects one category of the 3-class test fixture to treat as a binary classifier.
    binary_logits = logits[:, 1]
    binary_labels = labels == 1
    temperature_scaling = calibration.TemperatureScaling(binary=True)
    calibration_result = temperature_scaling.train_calibration(binary_logits, binary_labels)
    # Checks that we got close to optimal temperature
    assert temperature_scaling.temperature.item() == pytest.approx(8.3, EPSILON)
    # Checks that negative log-likelhood and expected calibration error are the same or lower post-calibration.
    assert calibration_result.after_calibration_nll <= calibration_result.before_calibration_nll
    assert calibration_result.after_calibration_ece <= calibration_result.before_calibration_ece


def test_temperature_scaling_category(uncalibrated_logits_and_labels):
    logits, labels = uncalibrated_logits_and_labels
    temperature_scaling = calibration.TemperatureScaling(num_classes=logits.shape[-1])
    calibration_result = temperature_scaling.train_calibration(logits, labels)
    # Checks that we got close to optimal temperature
    assert temperature_scaling.temperature.item() == pytest.approx(19.1, EPSILON)
    # Checks that negative log-likelhood and expected calibration error are the same or lower post-calibration.
    assert calibration_result.after_calibration_nll <= calibration_result.before_calibration_nll
    assert calibration_result.after_calibration_ece <= calibration_result.before_calibration_ece


def test_matrix_scaling_category(uncalibrated_logits_and_labels):
    logits, labels = uncalibrated_logits_and_labels
    matrix_scaling = calibration.MatrixScaling(num_classes=logits.shape[-1])
    calibration_result = matrix_scaling.train_calibration(logits, labels)
    # Matrix scaling may not have a single optimum, so multiple runs could give different results.
    # In this case we don't check any specific values
    # Checks that negative log-likelhood and expected calibration error are the same or lower post-calibration.
    assert calibration_result.after_calibration_nll <= calibration_result.before_calibration_nll
    assert calibration_result.after_calibration_ece <= calibration_result.before_calibration_ece
