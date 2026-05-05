"""Regression tests for Dask metadata inference failures.

These tests guard against the pattern of calling `.map(fn)` on a Dask Series without a
`meta=` argument.  Without `meta=`, Dask tries to infer the output dtype by calling `fn`
on a dummy sample, which fails for functions that return Python lists or numpy arrays
(e.g. `len`, tokenisers, array transformers).

All tests use the Ray backend with `processor: dask` so that the code paths that were
broken are actually exercised.  They are marked `distributed` and will be skipped when
Ray is not installed.

See https://github.com/ludwig-ai/ludwig/issues/4142 (PR #4144).
"""

import numpy as np
import pandas as pd
import pytest

ray = pytest.importorskip("ray")  # noqa
dask = pytest.importorskip("dask")  # noqa

pytestmark = pytest.mark.distributed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
VOCAB = "the quick brown fox jumps over lazy dog cat sat mat hat bat rat big small".split()
RAY_DASK_BACKEND = {"type": "ray", "processor": {"type": "dask"}}


def _make_text_df(n: int = 400) -> pd.DataFrame:
    texts = [" ".join(RNG.choice(VOCAB, size=RNG.integers(6, 14))) for _ in range(n)]
    labels = RNG.integers(0, 3, size=n).tolist()
    return pd.DataFrame({"text": texts, "label": labels})


def _make_number_df(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {f"f{i}": RNG.standard_normal(n).astype(np.float32) for i in range(5)}
        | {"target": RNG.integers(0, 2, size=n).tolist()}
    )


def _make_vector_df(n: int = 400, vec_size: int = 8) -> pd.DataFrame:
    vecs = [" ".join(str(x) for x in RNG.standard_normal(vec_size).round(4)) for _ in range(n)]
    return pd.DataFrame({"vec": vecs, "label": RNG.integers(0, 2, size=n).tolist()})


def _train_smoke(config: dict, df: pd.DataFrame) -> None:
    """Train for 1 epoch and assert no exceptions."""
    from ludwig.api import LudwigModel

    LudwigModel(config, logging_level=30).train(dataset=df)


# ---------------------------------------------------------------------------
# Text feature — bare .map() in create_vocabulary and build_sequence_matrix
# ---------------------------------------------------------------------------


def test_dask_text_create_vocabulary_no_meta_error(ray_cluster_2cpu):
    """Regression: strings_utils.create_vocabulary must not call .map(len) without meta=.

    Before the fix, `processed_lines.map(len)` in `create_vocabulary` caused
    `ValueError: Metadata inference failed in map` when using the Dask engine.
    """
    df = _make_text_df()
    config = {
        "input_features": [
            {
                "name": "text",
                "type": "text",
                "encoder": {"type": "parallel_cnn"},
                "preprocessing": {"max_sequence_length": 16},
            }
        ],
        "output_features": [{"name": "label", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    _train_smoke(config, df)


def test_dask_text_build_sequence_matrix_no_meta_error(ray_cluster_2cpu):
    """Regression: strings_utils.build_sequence_matrix must use processor.map_objects, not .map().

    Before the fix, `unit_vectors.map(len)` and the lambda map in `build_sequence_matrix`
    caused `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_text_df()
    config = {
        "input_features": [
            {
                "name": "text",
                "type": "text",
                "encoder": {"type": "stacked_cnn"},
                "preprocessing": {"max_sequence_length": 16},
            }
        ],
        "output_features": [{"name": "label", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    _train_smoke(config, df)


# ---------------------------------------------------------------------------
# Vector feature — bare .map(len) in get_feature_meta
# ---------------------------------------------------------------------------


def test_dask_vector_get_feature_meta_no_meta_error(ray_cluster_2cpu):
    """Regression: vector_feature.get_feature_meta must not call .map(len) without meta=.

    Before the fix the `.map(len).max()` call in `get_feature_meta` raised
    `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_vector_df()
    config = {
        "input_features": [{"name": "vec", "type": "vector"}],
        "output_features": [{"name": "label", "type": "binary"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    _train_smoke(config, df)


# ---------------------------------------------------------------------------
# Number features — verify that basic tabular Dask path still works
# ---------------------------------------------------------------------------


def test_dask_number_features_train(ray_cluster_2cpu):
    """Sanity check: tabular number features train without errors under Dask."""
    df = _make_number_df()
    config = {
        "input_features": [{"name": f"f{i}", "type": "number"} for i in range(5)],
        "output_features": [{"name": "target", "type": "binary"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    _train_smoke(config, df)


# ---------------------------------------------------------------------------
# Category output — bare .map() in postprocess_predictions
# ---------------------------------------------------------------------------


def test_dask_category_postprocess_no_meta_error(ray_cluster_2cpu):
    """Regression: category_feature.postprocess_predictions must use meta= on all .map() calls.

    Before the fix, `predictions[predictions_col].map(lambda pred: metadata["idx2str"][pred])`,
    `predictions[probabilities_col].map(max)`, and similar calls raised
    `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_text_df()
    config = {
        "input_features": [{"name": "text", "type": "text", "encoder": {"type": "parallel_cnn"}}],
        "output_features": [{"name": "label", "type": "category"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    from ludwig.api import LudwigModel

    model = LudwigModel(config, logging_level=30)
    _, _, output_dir = model.train(dataset=df)
    preds, _ = model.predict(dataset=df)
    assert preds is not None


# ---------------------------------------------------------------------------
# Binary output — bare .map() in postprocess_predictions
# ---------------------------------------------------------------------------


def _make_binary_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    texts = [" ".join(RNG.choice(VOCAB, size=rng.integers(6, 14))) for _ in range(n)]
    labels = rng.integers(0, 2, size=n).tolist()
    return pd.DataFrame({"text": texts, "label": labels})


def test_dask_binary_postprocess_no_meta_error(ray_cluster_2cpu):
    """Regression: binary_feature.postprocess_predictions must use meta= on .map() calls.

    Before the fix, `result[predictions_col].map(lambda pred: metadata["bool2str"][pred])` and
    `result[probabilities_col].map(lambda x: [1 - x, x])` raised
    `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_binary_df()
    config = {
        "input_features": [{"name": "text", "type": "text", "encoder": {"type": "parallel_cnn"}}],
        "output_features": [{"name": "label", "type": "binary"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    from ludwig.api import LudwigModel

    model = LudwigModel(config, logging_level=30)
    _, _, output_dir = model.train(dataset=df)
    preds, _ = model.predict(dataset=df)
    assert preds is not None


# ---------------------------------------------------------------------------
# Sequence output — bare .map() in postprocess_predictions
# ---------------------------------------------------------------------------


def _make_sequence_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # Input: space-separated token sequences; output: another sequence (seq2seq toy)
    inputs = [" ".join(RNG.choice(VOCAB, size=rng.integers(4, 10))) for _ in range(n)]
    targets = [" ".join(RNG.choice(VOCAB, size=rng.integers(3, 7))) for _ in range(n)]
    return pd.DataFrame({"src": inputs, "tgt": targets})


def test_dask_sequence_postprocess_no_meta_error(ray_cluster_2cpu):
    """Regression: sequence_feature.postprocess_predictions must use meta= on .map() calls.

    Before the fix, `result[last_preds_col].map(last_idx2str)` and
    `result[probs_col].map(compute_token_probabilities)` raised
    `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_sequence_df()
    config = {
        "input_features": [{"name": "src", "type": "sequence", "encoder": {"type": "embed"}}],
        "output_features": [{"name": "tgt", "type": "sequence"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    from ludwig.api import LudwigModel

    model = LudwigModel(config, logging_level=30)
    _, _, output_dir = model.train(dataset=df)
    preds, _ = model.predict(dataset=df)
    assert preds is not None


# ---------------------------------------------------------------------------
# Timeseries — bare .map(len) in build_matrix and bare .map() in postprocess
# ---------------------------------------------------------------------------


def _make_timeseries_df(n: int = 400, ts_len: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    ts = [" ".join(str(round(x, 4)) for x in rng.standard_normal(ts_len)) for _ in range(n)]
    targets = rng.standard_normal((n, ts_len)).round(4)
    target_strs = [" ".join(str(x) for x in row) for row in targets]
    return pd.DataFrame({"ts_in": ts, "ts_out": target_strs})


def test_dask_timeseries_no_meta_error(ray_cluster_2cpu):
    """Regression: timeseries_feature must use meta= on .map(len) and .map(lambda pred: pred.tolist()).

    Before the fix, `ts_vectors.map(len).max()` in `build_matrix` and
    `result[predictions_col].map(lambda pred: pred.tolist())` in `postprocess_predictions`
    raised `ValueError: Metadata inference failed in map` with the Dask engine.
    """
    df = _make_timeseries_df()
    config = {
        "input_features": [{"name": "ts_in", "type": "timeseries"}],
        "output_features": [{"name": "ts_out", "type": "timeseries"}],
        "trainer": {"epochs": 1, "batch_size": 64},
        "backend": RAY_DASK_BACKEND,
    }
    from ludwig.api import LudwigModel

    model = LudwigModel(config, logging_level=30)
    _, _, output_dir = model.train(dataset=df)
    preds, _ = model.predict(dataset=df)
    assert preds is not None
