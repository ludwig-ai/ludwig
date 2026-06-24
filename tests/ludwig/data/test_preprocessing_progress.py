import itertools
import time

import pandas as pd

from ludwig.callbacks import Callback
from ludwig.data.dataframe.pandas import PandasEngine
from ludwig.data.preprocessing_progress import get_total_partitions, PreprocessingProgressTracker


class ProgressCollector(Callback):
    def __init__(self):
        self.values = []

    def on_preprocess_progress(self, progress, **kwargs):
        self.values.append(progress)


def test_tracker_fires_progress_callbacks():
    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=3, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    for _ in range(3):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    time.sleep(0.7)
    tracker.stop()

    assert len(collector.values) > 0
    assert collector.values[-1] == 1.0


def test_tracker_no_callbacks_does_not_crash():
    tracker = PreprocessingProgressTracker(total=2, callbacks=[], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    engine.map_partitions(series, lambda s: s, progress_tracker=tracker)
    engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()


def test_map_partitions_without_tracker_unchanged():
    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    result = engine.map_partitions(series, lambda s: s * 2)
    pd.testing.assert_series_equal(result, series * 2)


def test_get_total_partitions_pandas():
    cols = {"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])}
    assert get_total_partitions(cols, use_ray=False) == 1


def test_progress_monotonically_increases():
    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=5, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series(range(10))
    for _ in range(5):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)
        time.sleep(0.1)

    tracker.stop()

    # Exclude the final forced 1.0 for monotonicity check on intermediate values
    intermediate = [v for v in collector.values if v < 1.0]
    for a, b in itertools.pairwise(intermediate):
        assert b >= a, f"Progress went backwards: {a} -> {b}"
    assert collector.values[-1] == 1.0
