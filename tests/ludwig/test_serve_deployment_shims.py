"""Phase 6.8 production deployment shims — unit tests.

These tests avoid actually starting Ray Serve or KServe (both are heavy optional dependencies).  They verify the import-
error contract when the optional package is missing and exercise the pure-Python helpers (v2 input transpose, CLI
argparse).
"""

from __future__ import annotations

import pytest

from ludwig.serve_kserve import _v2_inputs_to_records


class TestV2InputsTransposer:
    def test_basic_transpose(self):
        inputs = [
            {"name": "text", "shape": [2], "datatype": "BYTES", "data": ["hi", "bye"]},
            {"name": "num", "shape": [2], "datatype": "INT64", "data": [1, 2]},
        ]
        rows = _v2_inputs_to_records(inputs)
        assert rows == [{"text": "hi", "num": 1}, {"text": "bye", "num": 2}]

    def test_single_input(self):
        inputs = [{"name": "x", "shape": [3], "datatype": "FP32", "data": [0.1, 0.2, 0.3]}]
        rows = _v2_inputs_to_records(inputs)
        assert rows == [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}]

    def test_empty_inputs(self):
        assert _v2_inputs_to_records([]) == []

    def test_inconsistent_lengths_rejected(self):
        inputs = [
            {"name": "a", "shape": [2], "data": [1, 2]},
            {"name": "b", "shape": [3], "data": [1, 2, 3]},
        ]
        with pytest.raises(ValueError, match="inconsistent lengths"):
            _v2_inputs_to_records(inputs)


class TestRaySrveImportMissing:
    """If ray.serve isn't installed, the helpers raise a clear ImportError."""

    def test_deploy_helper_raises_when_ray_serve_missing(self):
        try:
            import ray.serve  # noqa: F401
        except ImportError:
            from ludwig.serve_ray_serve import deploy_ludwig_model

            with pytest.raises(ImportError, match="ludwig\\[distributed\\]"):
                deploy_ludwig_model(model_path="/does/not/matter", name="test")
        else:
            pytest.skip("ray.serve is installed; skipping missing-package test")


class TestKServeImportMissing:
    def test_serve_helper_raises_when_kserve_missing(self):
        try:
            import kserve  # noqa: F401
        except ImportError:
            from ludwig.serve_kserve import serve_ludwig_model

            with pytest.raises(ImportError, match="pip install kserve"):
                serve_ludwig_model("name", "/does/not/matter")
        else:
            pytest.skip("kserve is installed; skipping missing-package test")


class TestKServeCLIArgparse:
    def test_cli_requires_model_name_and_path(self):
        from ludwig.serve_kserve import cli

        with pytest.raises(SystemExit):
            cli([])  # argparse exits with code 2 on missing required args

    def test_cli_passes_parsed_args_to_server(self, monkeypatch):
        """Verify argparse extracts the right fields and delegates to serve_ludwig_model."""
        from ludwig import serve_kserve as module_under_test

        calls = {}

        def _fake_serve(model_name, model_path, http_port=8080):
            calls["model_name"] = model_name
            calls["model_path"] = model_path
            calls["http_port"] = http_port

        monkeypatch.setattr(module_under_test, "serve_ludwig_model", _fake_serve)
        module_under_test.cli(["--model_name", "m", "--model_path", "/p", "--http_port", "9000"])
        assert calls == {"model_name": "m", "model_path": "/p", "http_port": 9000}
