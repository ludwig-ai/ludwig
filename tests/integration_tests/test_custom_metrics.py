from collections import namedtuple

import numpy as np
import pytest

from ludwig.modules.metric_modules import R2Score, ErrorScore, JaccardMetric

RANDOM_SEED = 42
NUMBER_OBSERVATIONS = 500
SPLIT_POINT = NUMBER_OBSERVATIONS // 2

GeneratedData = namedtuple('GeneratedData',
                           'y_true y_good y_bad')


@pytest.fixture()
def generated_data():
    np.random.seed(RANDOM_SEED)

    # generate synthetic true values
    y_true = np.array(range(NUMBER_OBSERVATIONS)).astype(np.float32).reshape(
        -1, 1)

    # generate synthetic good predictions
    y_good = y_true + np.random.normal(size=y_true.shape[0]).reshape(-1, 1)

    # generate synthetic bad predictions
    y_bad = y_true + 146 * np.random.normal(size=y_true.shape[0]).reshape(-1,
                                                                          1)

    return GeneratedData(y_true=y_true, y_good=y_good, y_bad=y_bad)


def test_R2Score(generated_data):
    r2_score = R2Score()

    assert np.isnan(r2_score.result().numpy())

    # test as single batch
    r2_score.update_state(generated_data.y_true, generated_data.y_good)
    good_single_batch = r2_score.result().numpy()
    assert np.isreal(good_single_batch)

    # test as two batches
    r2_score.reset_states()
    r2_score.update_state(generated_data.y_true[:SPLIT_POINT],
                          generated_data.y_good[:SPLIT_POINT])
    r2_score.update_state(generated_data.y_true[SPLIT_POINT:],
                          generated_data.y_good[SPLIT_POINT:])
    good_two_batch = r2_score.result().numpy()
    assert np.isreal(good_two_batch)

    # single batch and multi-batch should be very close
    assert np.isclose(good_single_batch, good_two_batch)

    # good predictions should be close to 1
    assert np.abs(1.0 - good_two_batch) < 1e-3

    # test for bad predictions
    r2_score.reset_states()
    r2_score.update_state(generated_data.y_true[:SPLIT_POINT],
                          generated_data.y_bad[:SPLIT_POINT])
    r2_score.update_state(generated_data.y_true[SPLIT_POINT:],
                          generated_data.y_bad[SPLIT_POINT:])
    bad_prediction_score = r2_score.result().numpy()

    # r2 score for bad should be "far away" from 1
    assert bad_prediction_score < 0.05


def test_ErrorScore(generated_data):
    error_score = ErrorScore()

    assert np.isnan(error_score.result().numpy())

    # test as single batch
    error_score.update_state(generated_data.y_true, generated_data.y_good)
    good_single_batch = error_score.result().numpy()
    assert np.isreal(good_single_batch)

    # test as two batches
    error_score.reset_states()
    error_score.update_state(generated_data.y_true[:SPLIT_POINT],
                             generated_data.y_good[:SPLIT_POINT])
    error_score.update_state(generated_data.y_true[SPLIT_POINT:],
                             generated_data.y_good[SPLIT_POINT:])
    good_two_batch = error_score.result().numpy()
    assert np.isreal(good_two_batch)

    # single batch and multi-batch should be very close
    assert np.isclose(good_single_batch, good_two_batch)

    # test for bad predictions
    error_score.reset_states()
    error_score.update_state(generated_data.y_true[:SPLIT_POINT],
                             generated_data.y_bad[:SPLIT_POINT])
    error_score.update_state(generated_data.y_true[SPLIT_POINT:],
                             generated_data.y_bad[SPLIT_POINT:])
    bad_prediction_score = error_score.result().numpy()

    # magnitude of bad predictions should be greater than good predictions
    assert np.abs(bad_prediction_score) > np.abs(good_two_batch)


def test_JaccardMetric():
    # set up synthentic data for testing
    targets = np.array([
        [True, True, False], [True, False, True], [False, True, False],
        [True, True, True], [False, False, True]
    ])
    preds = np.array([
        [1, 0, 0], [0, 1, 1], [0, 1, 0],
        [1, 1, 1], [0, 0, 1]
    ])

    # create instance of jaccard metric object
    jm = JaccardMetric()

    # first test all the data at once
    jm.update_state(targets, preds)
    jaccard_big_batch = jm.result().numpy()

    # reset object and test multiple matches
    jm.reset_states()

    # Do two mini-batches
    jm.update_state(targets[:3], preds[:3])
    jm.update_state(targets[3:], preds[3:])
    jaccard_multiple_batches = jm.result().numpy()

    # check to make sure answers same for both big batch and multiple batches
    assert np.isclose(jaccard_big_batch, jaccard_multiple_batches)
