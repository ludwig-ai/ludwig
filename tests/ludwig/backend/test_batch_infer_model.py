"""Unit tests for the BatchInferModel inner class in ludwig.backend.ray.

BatchInferModel is a dynamically-generated class returned by
RayPredictor.get_batch_infer_model(). These tests exercise:
  - _prepare_batch: numpy conversion, stacking of non-scalar columns, reshape
without requiring a live Ray cluster, GPU, or real model weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ludwig.constants import NAME, PROC_COLUMN, TYPE

# BINARY/CATEGORY/NUMBER are scalar; other types (e.g. IMAGE) are non-scalar.
SCALAR_TYPE = "binary"
NON_SCALAR_TYPE = "image"


def _build_features(*col_defs):
    """Build (features, training_set_metadata) dicts from (col, type, reshape?) tuples."""
    features = {}
    metadata = {}
    for col_type_reshape in col_defs:
        col, col_type = col_type_reshape[:2]
        reshape = col_type_reshape[2] if len(col_type_reshape) > 2 else None
        features[col] = {TYPE: col_type, NAME: col, PROC_COLUMN: col}
        metadata[col] = {"reshape": reshape}
    return features, metadata


def _get_batch_infer_class(features, training_set_metadata, num_gpus=0):
    """Return the BatchInferModel class with all heavy dependencies mocked."""
    mock_model = MagicMock()
    mock_model.type.return_value = "ecd"
    mock_model.to.return_value = mock_model

    mock_predictor = MagicMock()

    with patch("ludwig.backend.ray.ray") as mock_ray:
        mock_ray.put.return_value = "obj_ref"
        mock_ray.get.return_value = mock_model

        with patch("ludwig.backend.ray.get_predictor_cls") as mock_get_cls:
            mock_get_cls.return_value = lambda **kw: mock_predictor

            with patch("ludwig.backend.ray.get_torch_device", return_value="cpu"):
                from ludwig.backend.ray import RayPredictor

                predictor = MagicMock(spec=RayPredictor)
                predictor.get_resources_per_worker = MagicMock(return_value=(1, num_gpus))

                BatchInferModel = RayPredictor.get_batch_infer_model(
                    predictor,
                    model=mock_model,
                    predictor_kwargs={},
                    output_columns=list(features.keys()),
                    features=features,
                    training_set_metadata=training_set_metadata,
                )

    return BatchInferModel, mock_model, mock_predictor


def _instantiate(BatchInferModel, mock_model, mock_predictor):
    """Instantiate BatchInferModel with mocked Ray and predictor."""
    with patch("ludwig.backend.ray.ray") as mock_ray:
        mock_ray.get.return_value = mock_model

        with patch("ludwig.backend.ray.get_predictor_cls") as mock_get_cls:
            mock_get_cls.return_value = lambda **kw: mock_predictor

            with patch("ludwig.backend.ray.get_torch_device", return_value="cpu"):
                return BatchInferModel()


class TestPrepareBatch:
    def test_scalar_column_converted_to_numpy(self):
        features, meta = _build_features(("label", SCALAR_TYPE))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        result = inst._prepare_batch(df)

        assert isinstance(result["label"], np.ndarray)
        np.testing.assert_array_equal(result["label"], [0, 1, 0, 1])

    def test_non_scalar_column_is_stacked(self):
        features, meta = _build_features(("image", NON_SCALAR_TYPE))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        rows = [np.zeros((3, 4)), np.ones((3, 4)), np.full((3, 4), 2.0)]
        df = pd.DataFrame({"image": rows})
        result = inst._prepare_batch(df)

        assert isinstance(result["image"], np.ndarray)
        assert result["image"].shape == (3, 3, 4)

    def test_reshape_applied(self):
        features, meta = _build_features(("flat", NON_SCALAR_TYPE, (2, 5)))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        rows = [np.zeros(10), np.ones(10)]
        df = pd.DataFrame({"flat": rows})
        result = inst._prepare_batch(df)

        assert result["flat"].shape == (2, 2, 5), result["flat"].shape

    def test_no_reshape_when_none(self):
        features, meta = _build_features(("x", NON_SCALAR_TYPE, None))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        rows = [np.zeros((4,)), np.ones((4,))]
        df = pd.DataFrame({"x": rows})
        result = inst._prepare_batch(df)

        assert result["x"].shape == (2, 4)

    def test_mixed_scalar_and_non_scalar(self):
        features, meta = _build_features(("cat", SCALAR_TYPE), ("img", NON_SCALAR_TYPE))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        df = pd.DataFrame(
            {
                "cat": [0, 1, 2],
                "img": [np.zeros((2,)), np.ones((2,)), np.full((2,), 3.0)],
            }
        )
        result = inst._prepare_batch(df)

        assert result["cat"].shape == (3,)
        assert result["img"].shape == (3, 2)

    def test_empty_dataframe_returns_empty_arrays(self):
        features, meta = _build_features(("label", SCALAR_TYPE))
        Cls, model, predictor = _get_batch_infer_class(features, meta)
        inst = _instantiate(Cls, model, predictor)

        df = pd.DataFrame({"label": pd.Series([], dtype=float)})
        result = inst._prepare_batch(df)

        assert result["label"].shape == (0,)
