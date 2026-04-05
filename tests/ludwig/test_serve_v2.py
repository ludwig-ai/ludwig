"""Tests for modernized serving."""

from ludwig.serve_v2 import _numpy_safe, build_request_schema, build_response_schema, ModelManager


class TestNumpySafe:
    def test_int(self):
        import numpy as np

        assert _numpy_safe(np.int64(42)) == 42
        assert isinstance(_numpy_safe(np.int64(42)), int)

    def test_float(self):
        import numpy as np

        assert isinstance(_numpy_safe(np.float32(3.14)), float)

    def test_array(self):
        import numpy as np

        result = _numpy_safe(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        import numpy as np

        data = {"a": np.int64(1), "b": {"c": np.array([1.0, 2.0])}}
        result = _numpy_safe(data)
        assert result == {"a": 1, "b": {"c": [1.0, 2.0]}}

    def test_passthrough(self):
        assert _numpy_safe("hello") == "hello"
        assert _numpy_safe(42) == 42


class TestSchemaGeneration:
    def test_build_request_schema(self):
        config = {
            "input_features": [
                {"name": "age", "type": "number"},
                {"name": "name", "type": "text"},
            ]
        }
        schema = build_request_schema(config)
        assert "age" in schema.model_fields
        assert "name" in schema.model_fields

    def test_build_response_schema(self):
        config = {
            "output_features": [
                {"name": "price", "type": "number"},
                {"name": "category", "type": "category"},
            ]
        }
        schema = build_response_schema(config)
        assert "price_predictions" in schema.model_fields
        assert "category_predictions" in schema.model_fields


class TestModelManager:
    def test_initial_state(self):
        mm = ModelManager()
        assert mm.model is None
        assert mm.config is None

    def test_get_model_raises_when_not_loaded(self):
        import pytest

        mm = ModelManager()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            mm.get_model()
