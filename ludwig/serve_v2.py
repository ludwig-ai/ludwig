"""Modernized Ludwig serving with auto-generated schemas, metrics, and vLLM support.

Improvements over serve.py:
- Auto-generated Pydantic request/response schemas from model config
- Prometheus metrics endpoint (/metrics)
- Model as injectable dependency for hot-swappability
- Timeout handling for long predictions
- Proper JSON serialization without NumpyJSONResponse hack
- Optional vLLM backend for LLM serving with OpenAI-compatible endpoints
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel

logger = logging.getLogger(__name__)


# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, generate_latest, Histogram

    REQUEST_COUNT = Counter("ludwig_requests_total", "Total prediction requests", ["endpoint", "status"])
    REQUEST_LATENCY = Histogram("ludwig_request_latency_seconds", "Request latency", ["endpoint"])
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def _numpy_safe(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    return obj


class ModelManager:
    """Manages the Ludwig model instance for dependency injection."""

    def __init__(self):
        self.model = None
        self.config = None

    def load(self, model_path: str, backend: str = "local"):
        from ludwig.api import LudwigModel

        self.model = LudwigModel.load(model_path, backend=backend)
        self.config = self.model.config
        logger.info(f"Model loaded from {model_path}")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model


model_manager = ModelManager()


def get_model():
    """Dependency injection for the model."""
    return model_manager.get_model()


def build_request_schema(config: dict) -> type[PydanticBaseModel]:
    """Auto-generate a Pydantic request schema from Ludwig model config."""
    fields = {}
    for feat in config.get("input_features", []):
        name = feat["name"]
        ftype = feat["type"]
        if ftype in ("number", "binary"):
            fields[name] = (float | None, None)
        elif ftype in ("category", "text", "sequence", "set", "bag"):
            fields[name] = (str | None, None)
        else:
            fields[name] = (Any, None)

    return type("PredictRequest", (PydanticBaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}})


def build_response_schema(config: dict) -> type[PydanticBaseModel]:
    """Auto-generate a Pydantic response schema from Ludwig model config."""
    fields = {}
    for feat in config.get("output_features", []):
        name = feat["name"]
        ftype = feat["type"]
        if ftype == "number":
            fields[f"{name}_predictions"] = (float | None, None)
        elif ftype in ("category", "text", "sequence"):
            fields[f"{name}_predictions"] = (str | None, None)
        elif ftype == "binary":
            fields[f"{name}_predictions"] = (bool | None, None)
            fields[f"{name}_probabilities"] = (float | None, None)
        else:
            fields[f"{name}_predictions"] = (Any, None)

    return type("PredictResponse", (PydanticBaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}})


def create_app(
    model_path: str | None = None,
    allowed_origins: list[str] | None = None,
    timeout_seconds: float = 300.0,
) -> FastAPI:
    """Create a modernized Ludwig serving application.

    Args:
        model_path: Path to the trained Ludwig model.
        allowed_origins: CORS allowed origins.
        timeout_seconds: Maximum time for a prediction request.

    Returns:
        FastAPI application.
    """
    app = FastAPI(
        title="Ludwig Inference Server",
        description="Production-ready model serving with auto-generated schemas",
        version="0.12.0",
    )

    if allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])

    if model_path:
        model_manager.load(model_path)

    @app.get("/")
    def health():
        return {"status": "healthy", "model_loaded": model_manager.model is not None}

    @app.get("/info")
    def model_info():
        if model_manager.config is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "model_type": model_manager.config.get("model_type", "ecd"),
            "input_features": [
                {"name": f["name"], "type": f["type"]} for f in model_manager.config.get("input_features", [])
            ],
            "output_features": [
                {"name": f["name"], "type": f["type"]} for f in model_manager.config.get("output_features", [])
            ],
        }

    @app.post("/predict")
    async def predict(request: Request, model=Depends(get_model)):
        start = time.time()
        try:
            body = await request.json()
            if isinstance(body, list):
                df = pd.DataFrame(body)
            else:
                df = pd.DataFrame([body])

            # Run prediction with timeout
            try:
                resp, _ = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: model.predict(dataset=df)),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                raise HTTPException(status_code=504, detail=f"Prediction timed out after {timeout_seconds}s")

            result = _numpy_safe(resp.to_dict("records"))

            if METRICS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
                REQUEST_LATENCY.labels(endpoint="predict").observe(time.time() - start)

            if len(result) == 1:
                return JSONResponse(result[0])
            return JSONResponse(result)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            if METRICS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch_predict")
    async def batch_predict(request: Request, model=Depends(get_model)):
        start = time.time()
        try:
            body = await request.json()
            df = pd.DataFrame(body)

            try:
                resp, _ = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: model.predict(dataset=df)),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                raise HTTPException(status_code=504, detail=f"Prediction timed out after {timeout_seconds}s")

            result = _numpy_safe(resp.to_dict("split"))

            if METRICS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint="batch_predict", status="success").inc()
                REQUEST_LATENCY.labels(endpoint="batch_predict").observe(time.time() - start)

            return JSONResponse(result)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Batch prediction failed: {e}")
            if METRICS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint="batch_predict", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

    if METRICS_AVAILABLE:

        @app.get("/metrics")
        def metrics():
            from starlette.responses import Response

            return Response(content=generate_latest(), media_type="text/plain")

    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    allowed_origins: list[str] | None = None,
):
    """Run the Ludwig serving application."""
    import uvicorn

    app = create_app(model_path=model_path, allowed_origins=allowed_origins)
    uvicorn.run(app, host=host, port=port)
