"""KServe v2 predictor shim for a Ludwig model (Phase 6.8).

KServe (and the successor ``llm-d`` project for LLM serving on Kubernetes) uses a
standard :class:`kserve.Model` class and an HTTP protocol called the
`Open Inference Protocol (v2) <https://kserve.github.io/website/modelserving/data_plane/v2_protocol/>`_.
This module provides a thin :class:`LudwigKServeModel` that wraps a trained
``LudwigModel`` and can be served via::

    python -m ludwig.serve_kserve --model_name=sentiment --model_path=/path/to/model

or by constructing ``LudwigKServeModel`` manually and handing it to ``kserve.ModelServer``.

The shim keeps the Ludwig-side predict contract unchanged — input features are sent
either as named inputs (v2 protocol) or as a dict-of-dicts in the request body, and
predictions are returned in the v2 response envelope.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_kserve():
    try:
        import kserve
    except ImportError as exc:
        raise ImportError("kserve is required for ludwig.serve_kserve. Install with: pip install kserve") from exc
    return kserve


def _build_model_class():
    """Construct the ``LudwigKServeModel`` class lazily so the module imports cleanly even when ``kserve`` isn't
    installed (e.g., during unit tests of the schema side)."""
    kserve = _import_kserve()

    class LudwigKServeModel(kserve.Model):
        def __init__(self, name: str, model_path: str) -> None:
            super().__init__(name)
            self.model_path = model_path
            self.ready = False
            self._model = None

        def load(self) -> bool:
            from ludwig.api import LudwigModel

            self._model = LudwigModel.load(self.model_path)
            self.ready = True
            logger.info("KServe loaded Ludwig model from %s", self.model_path)
            return self.ready

        async def predict(self, payload: dict | Any, headers: dict | None = None) -> dict:
            import pandas as pd

            # v2 protocol: {"inputs": [{"name": ..., "shape": ..., "datatype": ..., "data": [...]}]}
            if isinstance(payload, dict) and "inputs" in payload:
                records = _v2_inputs_to_records(payload["inputs"])
            else:
                # Tolerant fallback: accept a simple dict or list of dicts.
                records = payload if isinstance(payload, list) else [payload]

            df = pd.DataFrame(records)
            preds, _ = self._model.predict(dataset=df)
            pred_records = preds.to_dict(orient="records")

            # Emit v2 response envelope.
            outputs = []
            if pred_records:
                for col in pred_records[0].keys():
                    outputs.append(
                        {
                            "name": col,
                            "shape": [len(pred_records)],
                            # All outputs serialised as BYTES (string) for simplicity.
                            # A future improvement could infer FP32/INT64 etc. from the
                            # Ludwig output feature type to comply more strictly with v2.
                            "datatype": "BYTES",
                            "data": [str(rec.get(col)) for rec in pred_records],
                        }
                    )
            return {
                "model_name": self.name,
                "outputs": outputs,
            }

    return LudwigKServeModel


def _v2_inputs_to_records(inputs: list[dict]) -> list[dict]:
    """Transpose a v2-protocol input list into per-row records for Ludwig.

    v2 inputs look like::
        [{"name": "text", "shape": [2], "datatype": "BYTES", "data": ["hi", "bye"]},
         {"name": "num",  "shape": [2], "datatype": "INT64", "data": [1, 2]}]

    Ludwig wants::
        [{"text": "hi", "num": 1}, {"text": "bye", "num": 2}]
    """
    if not inputs:
        return []
    names = [inp["name"] for inp in inputs]
    datas = [inp["data"] for inp in inputs]
    lengths = [len(d) for d in datas]
    if len(set(lengths)) != 1:
        raise ValueError(f"v2 inputs have inconsistent lengths: {dict(zip(names, lengths))}")
    n = lengths[0]
    return [{name: data[i] for name, data in zip(names, datas)} for i in range(n)]


def serve_ludwig_model(model_name: str, model_path: str, http_port: int = 8080) -> None:
    """Launch a blocking KServe ``ModelServer`` hosting a Ludwig model.

    Args:
        model_name: name the model is registered under in the v2 protocol (used in
            the ``/v2/models/{name}/infer`` path).
        model_path: path to the trained Ludwig model directory.
        http_port: HTTP port to bind. Default matches KServe's convention.
    """
    kserve = _import_kserve()
    model_cls = _build_model_class()
    model = model_cls(model_name, model_path)
    model.load()
    server = kserve.ModelServer(http_port=http_port)
    server.start([model])


def cli(argv: list[str] | None = None) -> None:
    """Thin CLI entry point matching ``python -m ludwig.serve_kserve`` usage."""
    import argparse

    parser = argparse.ArgumentParser(prog="ludwig.serve_kserve", description="Run a Ludwig model under KServe.")
    parser.add_argument("--model_name", required=True, help="Model name exposed under /v2/models/<name>/infer")
    parser.add_argument("--model_path", required=True, help="Path to trained Ludwig model directory")
    parser.add_argument("--http_port", type=int, default=8080, help="HTTP port to bind")
    args = parser.parse_args(argv)

    serve_ludwig_model(args.model_name, args.model_path, http_port=args.http_port)


if __name__ == "__main__":  # pragma: no cover
    cli()
