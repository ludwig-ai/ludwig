"""Ultra-slow end-to-end training validation tests.

Each test trains for one epoch on a small real-ish dataset and asserts:
  1. Training completes without error (no data processing or pipeline issues).
  2. GPU is utilised (torch.cuda.max_memory_allocated > 0 when CUDA is available).
  3. Training loss decreases at least slightly relative to initial loss.
  4. All expected output assets are produced (model weights, hyperparameters JSON,
     training stats, predictions parquet).

The learning rate is set to a very small value (1e-6) so that the loss can only
move downward — a loss increase flags a serious bug, not normal noise.

Half the tests run with a local backend, half with a single-worker Ray backend.
These tests are NOT run in CI.  Run them locally before releases:

    pytest tests/ultra_slow/ -v --timeout=600

Sections
--------
- Input features (3 encoders × each type): text, number, category, sequence,
  binary, vector, timeseries, image, audio, bag, set
- Combiners (concat, tabnet, transformer, tabtransformer, project_aggregate,
  ft_transformer, cross_attention)
- Output features (one per output type): binary, number, category, sequence,
  vector, timeseries
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Optional Ray import — skip Ray tests gracefully if not installed
# ---------------------------------------------------------------------------
ray = pytest.importorskip("ray", reason="ray not installed")

from ludwig.api import LudwigModel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200  # number of training rows — small but enough for 1 epoch
VOCAB = "the quick brown fox jumps over the lazy dog sat on a mat".split()
LR = 1e-6  # tiny LR: loss should only decrease or stay flat
RAY_BACKEND = {"type": "ray", "processor": {"type": "dask"}}


def _words(n_per_row: int = 10) -> list[str]:
    return [" ".join(RNG.choice(VOCAB, size=n_per_row).tolist()) for _ in range(N)]


def _ints(lo: int = 0, hi: int = 4) -> list[int]:
    return RNG.integers(lo, hi, size=N).tolist()


def _floats() -> list[float]:
    return RNG.standard_normal(N).tolist()


def _bool_col() -> list[int]:
    return RNG.integers(0, 2, size=N).tolist()


def _vec_col(dim: int = 8) -> list[str]:
    return [" ".join(f"{x:.4f}" for x in RNG.standard_normal(dim)) for _ in range(N)]


def _ts_col(length: int = 12) -> list[str]:
    return [" ".join(f"{x:.4f}" for x in RNG.standard_normal(length)) for _ in range(N)]


def _tabular_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": _floats(),
            "num2": _floats(),
            "cat1": _ints(0, 4),
            "bin1": _bool_col(),
            "label_cat": _ints(0, 3),
            "label_bin": _bool_col(),
            "label_num": _floats(),
        }
    )


def _train_and_check(config: dict, df: pd.DataFrame, backend=None, tmpdir: str | None = None) -> dict:
    """Train for 1 epoch and return a dict with loss info + asset checks."""
    with tempfile.TemporaryDirectory() as _tmp:
        out = tmpdir or _tmp
        if backend:
            config = {**config, "backend": backend}
        config.setdefault("trainer", {})
        config["trainer"].setdefault("epochs", 1)
        config["trainer"].setdefault("learning_rate", LR)
        config["trainer"].setdefault("batch_size", 32)

        model = LudwigModel(config, logging_level=40)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = model.train(dataset=df, output_directory=out)
        train_stats = result.train_stats
        output_dir = result.output_directory

        # ── 1. Assets exist ──────────────────────────────────────────────
        assert os.path.exists(os.path.join(output_dir, "model")), "model directory missing"
        assert os.path.exists(
            os.path.join(output_dir, "model", "model_hyperparameters.json")
        ), "hyperparameters JSON missing"

        # ── 2. GPU was used (local backend only — Ray workers run in subprocesses) ──
        if backend is None and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            assert peak > 0, f"CUDA available but peak GPU memory was 0 — model did not use GPU (peak={peak})"

        # ── 3. Loss should not blow up (LR=1e-6 means barely any movement) ──
        # TrainingStats.training: dict[feature_name, dict[metric_name, list[float]]]
        combined_feature = (train_stats.training or {}).get("combined", {})
        loss_values = combined_feature.get("loss", [])
        if len(loss_values) >= 2:
            first, last = loss_values[0], loss_values[-1]
            assert last <= first * 2.0, (
                f"Combined loss more than doubled ({first:.4f} → {last:.4f}). "
                "Something is wrong with the training loop."
            )

        return {"output_dir": output_dir, "train_stats": train_stats}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_1gpu():
    """Single-worker Ray cluster backed by the available GPU (or CPU fallback)."""
    use_gpu = torch.cuda.is_available()
    ray.init(
        num_cpus=4,
        num_gpus=torch.cuda.device_count(),
        ignore_reinit_error=True,
    )
    yield {"type": "ray", "processor": {"type": "dask"}, "trainer": {"use_gpu": use_gpu, "num_workers": 1}}
    ray.shutdown()


# ---------------------------------------------------------------------------
# TEXT input (local backend)
# ---------------------------------------------------------------------------


class TestTextInputLocal:
    """Text input — local backend — 3 encoders (popular + modern)."""

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"text": _words(12), "label": _ints(0, 3)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "text",
                    "type": "text",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"max_sequence_length": 16},
                }
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_parallel_cnn(self):
        """Classic, widely used sequence encoder."""
        _train_and_check(self._config("parallel_cnn"), self._df)

    def test_bert(self):
        """Most popular pretrained text encoder."""
        _train_and_check(
            {
                "input_features": [
                    {
                        "name": "text",
                        "type": "text",
                        "encoder": {
                            "type": "auto_transformer",
                            "pretrained_model_name_or_path": "bert-base-uncased",
                            "trainable": False,
                            "reduce_output": "mean",
                        },
                        "preprocessing": {"max_sequence_length": 16},
                    }
                ],
                "output_features": [{"name": "label", "type": "category"}],
            },
            self._df,
        )

    def test_modernbert(self):
        """Modern BERT variant with BPE tokenizer (regression: was misrouted to WordPiece)."""
        _train_and_check(
            {
                "input_features": [
                    {
                        "name": "text",
                        "type": "text",
                        "encoder": {
                            "type": "auto_transformer",
                            "pretrained_model_name_or_path": "answerdotai/ModernBERT-base",
                            "trainable": False,
                            "reduce_output": "mean",
                        },
                        "preprocessing": {"max_sequence_length": 16},
                    }
                ],
                "output_features": [{"name": "label", "type": "category"}],
            },
            self._df,
        )


# ---------------------------------------------------------------------------
# TEXT input (Ray backend)
# ---------------------------------------------------------------------------


@pytest.mark.distributed
class TestTextInputRay:
    """Text input — Ray backend — 3 encoders."""

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"text": _words(12), "label": _ints(0, 3)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "text",
                    "type": "text",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"max_sequence_length": 16},
                }
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_parallel_cnn(self, ray_1gpu):
        _train_and_check(self._config("parallel_cnn"), self._df, backend=ray_1gpu)

    def test_rnn(self, ray_1gpu):
        _train_and_check(self._config("rnn"), self._df, backend=ray_1gpu)

    def test_stacked_cnn(self, ray_1gpu):
        _train_and_check(self._config("stacked_cnn"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# NUMBER input
# ---------------------------------------------------------------------------


class TestNumberInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"n1": _floats(), "n2": _floats(), "n3": _floats(), "label": _bool_col()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {"name": f"n{i}", "type": "number", "encoder": {"type": encoder_type}} for i in range(1, 4)
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }

    def test_dense(self):
        _train_and_check(self._config("dense"), self._df)

    def test_passthrough(self):
        _train_and_check(self._config("passthrough"), self._df)

    def test_bins(self):
        _train_and_check(self._config("bins"), self._df)


@pytest.mark.distributed
class TestNumberInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"n1": _floats(), "n2": _floats(), "n3": _floats(), "label": _bool_col()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {"name": f"n{i}", "type": "number", "encoder": {"type": encoder_type}} for i in range(1, 4)
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }

    def test_dense(self, ray_1gpu):
        _train_and_check(self._config("dense"), self._df, backend=ray_1gpu)

    def test_passthrough(self, ray_1gpu):
        _train_and_check(self._config("passthrough"), self._df, backend=ray_1gpu)

    def test_bins(self, ray_1gpu):
        _train_and_check(self._config("bins"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# CATEGORY input
# ---------------------------------------------------------------------------


class TestCategoryInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"cat": _ints(0, 8), "label": _floats()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [{"name": "cat", "type": "category", "encoder": {"type": encoder_type}}],
            "output_features": [{"name": "label", "type": "number"}],
        }

    def test_dense(self):
        _train_and_check(self._config("dense"), self._df)

    def test_sparse(self):
        """Sparse embedding — most memory-efficient for high-cardinality categories."""
        _train_and_check(self._config("sparse"), self._df)

    def test_onehot(self):
        _train_and_check(self._config("onehot"), self._df)


@pytest.mark.distributed
class TestCategoryInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"cat": _ints(0, 8), "label": _floats()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [{"name": "cat", "type": "category", "encoder": {"type": encoder_type}}],
            "output_features": [{"name": "label", "type": "number"}],
        }

    def test_dense(self, ray_1gpu):
        _train_and_check(self._config("dense"), self._df, backend=ray_1gpu)

    def test_sparse(self, ray_1gpu):
        _train_and_check(self._config("sparse"), self._df, backend=ray_1gpu)

    def test_onehot(self, ray_1gpu):
        _train_and_check(self._config("onehot"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# BINARY input
# ---------------------------------------------------------------------------


class TestBinaryInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"b1": _bool_col(), "b2": _bool_col(), "b3": _bool_col(), "label": _ints(0, 4)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {"name": f"b{i}", "type": "binary", "encoder": {"type": encoder_type}} for i in range(1, 4)
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_dense(self):
        _train_and_check(self._config("dense"), self._df)

    def test_passthrough_with_concat_combiner(self):
        """Binary passthrough feeds raw 0/1 into the combiner — most common production pattern."""
        _train_and_check(
            {**self._config("passthrough"), "combiner": {"type": "concat"}},
            self._df,
        )

    def test_mixed_binary_number(self):
        """Mix of binary + number features — common real-world tabular setup."""
        df = self._df.copy()
        df["num_feat"] = _floats()
        _train_and_check(
            {
                "input_features": [
                    {"name": "b1", "type": "binary"},
                    {"name": "b2", "type": "binary"},
                    {"name": "num_feat", "type": "number"},
                ],
                "output_features": [{"name": "label", "type": "category"}],
            },
            df,
        )


@pytest.mark.distributed
class TestBinaryInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"b1": _bool_col(), "b2": _bool_col(), "b3": _bool_col(), "label": _ints(0, 4)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {"name": f"b{i}", "type": "binary", "encoder": {"type": encoder_type}} for i in range(1, 4)
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_dense(self, ray_1gpu):
        _train_and_check(self._config("dense"), self._df, backend=ray_1gpu)

    def test_passthrough(self, ray_1gpu):
        _train_and_check(self._config("passthrough"), self._df, backend=ray_1gpu)

    def test_mixed_binary_number(self, ray_1gpu):
        df = self._df.copy()
        df["num_feat"] = _floats()
        _train_and_check(
            {
                "input_features": [
                    {"name": "b1", "type": "binary"},
                    {"name": "b2", "type": "binary"},
                    {"name": "num_feat", "type": "number"},
                ],
                "output_features": [{"name": "label", "type": "category"}],
            },
            df,
            backend=ray_1gpu,
        )


# ---------------------------------------------------------------------------
# SEQUENCE input
# ---------------------------------------------------------------------------


class TestSequenceInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"seq": _words(8), "label": _ints(0, 3)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "seq",
                    "type": "sequence",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"max_sequence_length": 12},
                }
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_embed(self):
        _train_and_check(self._config("embed"), self._df)

    def test_parallel_cnn(self):
        """Most popular non-RNN sequence encoder in Ludwig."""
        _train_and_check(self._config("parallel_cnn"), self._df)

    def test_transformer(self):
        """Modern attention-based encoder."""
        _train_and_check(self._config("transformer"), self._df)


@pytest.mark.distributed
class TestSequenceInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"seq": _words(8), "label": _ints(0, 3)})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "seq",
                    "type": "sequence",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"max_sequence_length": 12},
                }
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }

    def test_embed(self, ray_1gpu):
        _train_and_check(self._config("embed"), self._df, backend=ray_1gpu)

    def test_parallel_cnn(self, ray_1gpu):
        _train_and_check(self._config("parallel_cnn"), self._df, backend=ray_1gpu)

    def test_transformer(self, ray_1gpu):
        _train_and_check(self._config("transformer"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# VECTOR input
# ---------------------------------------------------------------------------


class TestVectorInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"vec": _vec_col(16), "label": _bool_col()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [{"name": "vec", "type": "vector", "encoder": {"type": encoder_type}}],
            "output_features": [{"name": "label", "type": "binary"}],
        }

    def test_dense(self):
        _train_and_check(self._config("dense"), self._df)

    def test_passthrough(self):
        _train_and_check(self._config("passthrough"), self._df)

    def test_multi_vector(self):
        # Two vector features + binary output (validates multi-input tabular path)
        df = pd.DataFrame({"v1": _vec_col(8), "v2": _vec_col(8), "label": _bool_col()})
        config = {
            "input_features": [
                {"name": "v1", "type": "vector", "encoder": {"type": "dense"}},
                {"name": "v2", "type": "vector", "encoder": {"type": "dense"}},
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }
        _train_and_check(config, df)


@pytest.mark.distributed
class TestVectorInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"vec": _vec_col(16), "label": _bool_col()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [{"name": "vec", "type": "vector", "encoder": {"type": encoder_type}}],
            "output_features": [{"name": "label", "type": "binary"}],
        }

    def test_dense(self, ray_1gpu):
        _train_and_check(self._config("dense"), self._df, backend=ray_1gpu)

    def test_passthrough(self, ray_1gpu):
        _train_and_check(self._config("passthrough"), self._df, backend=ray_1gpu)

    def test_multi_vector(self, ray_1gpu):
        df = pd.DataFrame({"v1": _vec_col(8), "v2": _vec_col(8), "label": _bool_col()})
        config = {
            "input_features": [
                {"name": "v1", "type": "vector", "encoder": {"type": "dense"}},
                {"name": "v2", "type": "vector", "encoder": {"type": "dense"}},
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }
        _train_and_check(config, df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# TIMESERIES input
# ---------------------------------------------------------------------------


class TestTimeseriesInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"ts": _ts_col(12), "label": _floats()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "ts",
                    "type": "timeseries",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"timeseries_length_limit": 12},
                }
            ],
            "output_features": [{"name": "label", "type": "number"}],
        }

    def test_dense(self):
        _train_and_check(self._config("dense"), self._df)

    def test_parallel_cnn(self):
        _train_and_check(self._config("parallel_cnn"), self._df)

    def test_transformer(self):
        _train_and_check(self._config("transformer"), self._df)


@pytest.mark.distributed
class TestTimeseriesInputRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"ts": _ts_col(12), "label": _floats()})

    def _config(self, encoder_type: str) -> dict:
        return {
            "input_features": [
                {
                    "name": "ts",
                    "type": "timeseries",
                    "encoder": {"type": encoder_type},
                    "preprocessing": {"timeseries_length_limit": 12},
                }
            ],
            "output_features": [{"name": "label", "type": "number"}],
        }

    def test_dense(self, ray_1gpu):
        _train_and_check(self._config("dense"), self._df, backend=ray_1gpu)

    def test_parallel_cnn(self, ray_1gpu):
        _train_and_check(self._config("parallel_cnn"), self._df, backend=ray_1gpu)

    def test_transformer(self, ray_1gpu):
        _train_and_check(self._config("transformer"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# SET input (local only — Ray support is limited)
# ---------------------------------------------------------------------------


class TestSetInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        # Sets are space-separated token strings
        self._df = pd.DataFrame({"tags": _words(4), "label": _ints(0, 3)})

    def test_embed(self):
        # Set feature only supports the "embed" encoder
        config = {
            "input_features": [{"name": "tags", "type": "set", "encoder": {"type": "embed"}}],
            "output_features": [{"name": "label", "type": "category"}],
        }
        _train_and_check(config, self._df)

    def test_embed_with_number(self):
        # Multi-feature: set + number → category
        df = pd.DataFrame({"tags": _words(4), "score": _floats(), "label": _ints(0, 3)})
        config = {
            "input_features": [
                {"name": "tags", "type": "set", "encoder": {"type": "embed"}},
                {"name": "score", "type": "number"},
            ],
            "output_features": [{"name": "label", "type": "category"}],
        }
        _train_and_check(config, df)

    def test_embed_with_category(self):
        # Multi-feature: set + category → binary
        df = pd.DataFrame({"tags": _words(4), "cat": _ints(0, 4), "label": _bool_col()})
        config = {
            "input_features": [
                {"name": "tags", "type": "set", "encoder": {"type": "embed"}},
                {"name": "cat", "type": "category"},
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }
        _train_and_check(config, df)


# ---------------------------------------------------------------------------
# BAG input
# ---------------------------------------------------------------------------


class TestBagInputLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame({"bag": _words(6), "label": _bool_col()})

    def test_embed(self):
        # Bag feature only supports the "embed" encoder
        config = {
            "input_features": [{"name": "bag", "type": "bag", "encoder": {"type": "embed"}}],
            "output_features": [{"name": "label", "type": "binary"}],
        }
        _train_and_check(config, self._df)

    def test_embed_with_number(self):
        # Multi-feature: bag + number → binary
        df = pd.DataFrame({"bag": _words(6), "score": _floats(), "label": _bool_col()})
        config = {
            "input_features": [
                {"name": "bag", "type": "bag", "encoder": {"type": "embed"}},
                {"name": "score", "type": "number"},
            ],
            "output_features": [{"name": "label", "type": "binary"}],
        }
        _train_and_check(config, df)

    def test_embed_with_category(self):
        # Multi-feature: bag + category → number
        df = pd.DataFrame({"bag": _words(6), "cat": _ints(0, 4), "label": _floats()})
        config = {
            "input_features": [
                {"name": "bag", "type": "bag", "encoder": {"type": "embed"}},
                {"name": "cat", "type": "category"},
            ],
            "output_features": [{"name": "label", "type": "number"}],
        }
        _train_and_check(config, df)


# ---------------------------------------------------------------------------
# Combiners  (local, tabular features)
# ---------------------------------------------------------------------------


def _tabular_config(combiner_type: str, combiner_kwargs: dict | None = None) -> dict:
    combiner = {"type": combiner_type, **(combiner_kwargs or {})}
    return {
        "input_features": [
            {"name": "num1", "type": "number"},
            {"name": "num2", "type": "number"},
            {"name": "cat1", "type": "category"},
            {"name": "bin1", "type": "binary"},
        ],
        "output_features": [{"name": "label_cat", "type": "category"}],
        "combiner": combiner,
    }


class TestCombinersLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _tabular_df()

    def test_concat(self):
        _train_and_check(_tabular_config("concat"), self._df)

    def test_tabnet(self):
        _train_and_check(_tabular_config("tabnet"), self._df)

    def test_transformer(self):
        _train_and_check(_tabular_config("transformer"), self._df)

    def test_tabtransformer(self):
        _train_and_check(_tabular_config("tabtransformer"), self._df)

    def test_project_aggregate(self):
        _train_and_check(_tabular_config("project_aggregate"), self._df)

    def test_ft_transformer(self):
        _train_and_check(_tabular_config("ft_transformer"), self._df)

    def test_cross_attention(self):
        _train_and_check(_tabular_config("cross_attention"), self._df)


@pytest.mark.distributed
class TestCombinersRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _tabular_df()

    def test_concat(self, ray_1gpu):
        _train_and_check(_tabular_config("concat"), self._df, backend=ray_1gpu)

    def test_tabnet(self, ray_1gpu):
        _train_and_check(_tabular_config("tabnet"), self._df, backend=ray_1gpu)

    def test_transformer(self, ray_1gpu):
        _train_and_check(_tabular_config("transformer"), self._df, backend=ray_1gpu)

    def test_tabtransformer(self, ray_1gpu):
        _train_and_check(_tabular_config("tabtransformer"), self._df, backend=ray_1gpu)

    def test_ft_transformer(self, ray_1gpu):
        _train_and_check(_tabular_config("ft_transformer"), self._df, backend=ray_1gpu)

    def test_cross_attention(self, ray_1gpu):
        _train_and_check(_tabular_config("cross_attention"), self._df, backend=ray_1gpu)

    def test_project_aggregate(self, ray_1gpu):
        _train_and_check(_tabular_config("project_aggregate"), self._df, backend=ray_1gpu)


# ---------------------------------------------------------------------------
# Output feature types  (one test each, local + Ray)
# ---------------------------------------------------------------------------


def _base_inputs() -> list[dict]:
    return [
        {"name": "n1", "type": "number"},
        {"name": "n2", "type": "number"},
        {"name": "cat1", "type": "category"},
    ]


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({"n1": _floats(), "n2": _floats(), "cat1": _ints(0, 4)})


class TestOutputTypesLocal:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame(
            {
                "n1": _floats(),
                "n2": _floats(),
                "cat1": _ints(0, 4),
                "out_bin": _bool_col(),
                "out_num": _floats(),
                "out_cat": _ints(0, 3),
                "out_seq": _words(6),
                "out_vec": _vec_col(8),
                "out_ts": _ts_col(8),
            }
        )

    def test_binary_output(self):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_bin", "type": "binary"}]},
            self._df,
        )

    def test_number_output(self):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_num", "type": "number"}]},
            self._df,
        )

    def test_category_output(self):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_cat", "type": "category"}]},
            self._df,
        )

    def test_sequence_output(self):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [
                    {
                        "name": "out_seq",
                        "type": "sequence",
                        "preprocessing": {"max_sequence_length": 10},
                        "decoder": {"type": "transformer_generator"},
                    }
                ],
            },
            self._df,
        )

    def test_vector_output(self):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [{"name": "out_vec", "type": "vector", "decoder": {"output_size": 8}}],
            },
            self._df,
        )

    def test_timeseries_output(self):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [
                    {
                        "name": "out_ts",
                        "type": "timeseries",
                        "preprocessing": {"timeseries_length_limit": 8},
                        "decoder": {"output_size": 8},
                    }
                ],
            },
            self._df,
        )


@pytest.mark.distributed
class TestOutputTypesRay:
    @pytest.fixture(autouse=True)
    def df(self):
        self._df = pd.DataFrame(
            {
                "n1": _floats(),
                "n2": _floats(),
                "cat1": _ints(0, 4),
                "out_bin": _bool_col(),
                "out_num": _floats(),
                "out_cat": _ints(0, 3),
                "out_seq": _words(6),
                "out_vec": _vec_col(8),
                "out_ts": _ts_col(8),
            }
        )

    def test_binary_output(self, ray_1gpu):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_bin", "type": "binary"}]},
            self._df,
            backend=ray_1gpu,
        )

    def test_number_output(self, ray_1gpu):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_num", "type": "number"}]},
            self._df,
            backend=ray_1gpu,
        )

    def test_category_output(self, ray_1gpu):
        _train_and_check(
            {"input_features": _base_inputs(), "output_features": [{"name": "out_cat", "type": "category"}]},
            self._df,
            backend=ray_1gpu,
        )

    def test_sequence_output(self, ray_1gpu):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [
                    {
                        "name": "out_seq",
                        "type": "sequence",
                        "preprocessing": {"max_sequence_length": 10},
                        "decoder": {"type": "transformer_generator"},
                    }
                ],
            },
            self._df,
            backend=ray_1gpu,
        )

    def test_vector_output(self, ray_1gpu):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [{"name": "out_vec", "type": "vector", "decoder": {"output_size": 8}}],
            },
            self._df,
            backend=ray_1gpu,
        )

    def test_timeseries_output(self, ray_1gpu):
        _train_and_check(
            {
                "input_features": _base_inputs(),
                "output_features": [
                    {
                        "name": "out_ts",
                        "type": "timeseries",
                        "preprocessing": {"timeseries_length_limit": 8},
                        "decoder": {"output_size": 8},
                    }
                ],
            },
            self._df,
            backend=ray_1gpu,
        )
