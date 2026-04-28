"""Tests for LLM-driven config generation (mocked LLM calls)."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from ludwig.config_generation import generate_config, get_ludwig_schema_context


class TestGetLudwigSchemaContext:
    def test_returns_valid_json(self):
        context = get_ludwig_schema_context()
        parsed = json.loads(context)
        assert "input_feature_types" in parsed
        assert "combiner_types" in parsed
        assert "example_config" in parsed

    def test_contains_feature_types(self):
        context = get_ludwig_schema_context()
        parsed = json.loads(context)
        types = parsed["input_feature_types"]
        assert "number" in types
        assert "category" in types
        assert "text" in types
        assert "binary" in types


class TestGenerateConfig:
    VALID_CONFIG_JSON = json.dumps(
        {
            "input_features": [
                {"name": "age", "type": "number"},
                {"name": "income", "type": "number"},
            ],
            "output_features": [{"name": "churn", "type": "binary"}],
            "combiner": {"type": "concat"},
            "trainer": {"epochs": 50, "batch_size": 128, "learning_rate": 0.001},
        }
    )

    def _mock_anthropic(self, response_text):
        """Create a mock anthropic module with the given response text."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_module = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        return mock_module, mock_client

    def test_generate_config_anthropic(self):
        mock_module, mock_client = self._mock_anthropic(self.VALID_CONFIG_JSON)

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            config = generate_config(
                "Predict customer churn from age and income",
                validate=False,
            )

        assert "input_features" in config
        assert config["output_features"][0]["name"] == "churn"
        mock_client.messages.create.assert_called_once()

    def test_generate_config_strips_markdown_code_blocks(self):
        wrapped = f"```json\n{self.VALID_CONFIG_JSON}\n```"
        mock_module, _ = self._mock_anthropic(wrapped)

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            config = generate_config("some task", validate=False)
        assert "input_features" in config

    def test_generate_config_invalid_json_raises(self):
        mock_module, _ = self._mock_anthropic("not json at all")

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            with pytest.raises(json.JSONDecodeError):
                generate_config("some task", validate=False)
