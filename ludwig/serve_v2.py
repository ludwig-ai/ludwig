"""Modernized Ludwig serving with auto-generated schemas, metrics, and vLLM support.

Improvements over the original serve.py:
- Auto-generated Pydantic request/response schemas from model config
- Prometheus metrics endpoint (/metrics)
- Model as injectable dependency for hot-swappability
- Timeout handling for long predictions (prediction_timeout parameter)
- Proper JSON serialization without NumpyJSONResponse hack
- Structured key=value logging for every request and response
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
from pydantic import create_model, Field

from ludwig.constants import COLUMN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metrics (optional dependency)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest, Histogram

    _REQUEST_COUNT = Counter(
        "ludwig_requests_total",
        "Total prediction requests",
        ["endpoint", "status"],
    )
    _REQUEST_LATENCY = Histogram(
        "ludwig_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
    )
    _ERROR_COUNT = Counter(
        "ludwig_errors_total",
        "Total prediction errors",
        ["endpoint"],
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numpy → Python serialization helpers
# ---------------------------------------------------------------------------
def _numpy_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_numpy_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Auto-generated Pydantic schemas from model config
# ---------------------------------------------------------------------------
_FEATURE_TYPE_TO_PYTHON: dict[str, type] = {
    "number": float,
    "binary": bool,
    "category": str,
    "text": str,
    "sequence": str,
    "set": str,
    "bag": str,
    "date": str,
    "h3": str,
    "vector": list,
    "image": Any,
    "audio": Any,
    "timeseries": Any,
}


def _feature_python_type(ftype: str) -> type:
    return _FEATURE_TYPE_TO_PYTHON.get(ftype, Any)


def build_request_schema(config: dict) -> type[PydanticBaseModel]:
    """Dynamically create a Pydantic request model from Ludwig input feature configs.

    Each input feature becomes an optional field (None default) so that missing features can be detected and reported
    with a clear error message rather than a Pydantic validation error.
    """
    fields: dict[str, Any] = {}
    for feat in config.get("input_features", []):
        name = feat.get("name") or feat.get("column") or feat.get(COLUMN)
        ftype = feat.get("type", "")
        py_type = _feature_python_type(ftype)
        fields[name] = (py_type | None, Field(default=None, description=f"Input feature '{name}' of type '{ftype}'"))

    return create_model("PredictRequest", **fields)


def build_response_schema(config: dict) -> type[PydanticBaseModel]:
    """Dynamically create a Pydantic response model from Ludwig output feature configs.

    The exact column names produced by Ludwig's post-processor (e.g. ``age_predictions``,
    ``churn_probabilities``) are declared as optional fields so extra columns don't break
    validation.
    """
    fields: dict[str, Any] = {}
    for feat in config.get("output_features", []):
        name = feat.get("name") or feat.get("column") or feat.get(COLUMN)
        ftype = feat.get("type", "")

        if ftype == "number":
            fields[f"{name}_predictions"] = (float | None, None)
        elif ftype == "binary":
            fields[f"{name}_predictions"] = (bool | None, None)
            fields[f"{name}_probabilities"] = (float | None, None)
        elif ftype in ("category", "text", "sequence", "set", "bag", "date", "h3"):
            fields[f"{name}_predictions"] = (str | None, None)
            fields[f"{name}_probabilities"] = (list | None, None)
        else:
            fields[f"{name}_predictions"] = (Any, None)

    return create_model("PredictResponse", **fields)


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------
def _record_success(endpoint: str, latency: float) -> None:
    if METRICS_AVAILABLE:
        _REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        _REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


def _record_error(endpoint: str) -> None:
    if METRICS_AVAILABLE:
        _REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        _ERROR_COUNT.labels(endpoint=endpoint).inc()


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------
def _log_request(endpoint: str, feature_names: list[str], batch_size: int) -> None:
    logger.info(
        "prediction_request endpoint=%s features=%s batch_size=%d",
        endpoint,
        ",".join(feature_names),
        batch_size,
    )


def _log_response(endpoint: str, output_feature_names: list[str], latency: float) -> None:
    logger.info(
        "prediction_response endpoint=%s outputs=%s latency_seconds=%.4f",
        endpoint,
        ",".join(output_feature_names),
        latency,
    )


# ---------------------------------------------------------------------------
# Model manager (dependency injection)
# ---------------------------------------------------------------------------
class ModelManager:
    """Manages the Ludwig model instance for dependency injection."""

    def __init__(self):
        self.model = None
        self.config: dict | None = None
        self._input_feature_names: list[str] = []
        self._output_feature_names: list[str] = []
        self._input_features_set: set[str] = set()

    def load(self, model_path: str, backend: str = "local") -> None:
        from ludwig.api import LudwigModel

        self.model = LudwigModel.load(model_path, backend=backend)
        self.config = self.model.config
        self._input_feature_names = [
            f.get("name") or f.get("column") or f.get(COLUMN) for f in self.config.get("input_features", [])
        ]
        self._output_feature_names = [
            f.get("name") or f.get("column") or f.get(COLUMN) for f in self.config.get("output_features", [])
        ]
        self._input_features_set = set(self._input_feature_names)
        logger.info("model_loaded path=%s", model_path)

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model


model_manager = ModelManager()


def get_model():
    """FastAPI dependency — returns the loaded Ludwig model."""
    return model_manager.get_model()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    model_path: str | None = None,
    allowed_origins: list[str] | None = None,
    prediction_timeout: float = 30.0,
) -> FastAPI:
    """Create a modernized Ludwig serving application.

    Args:
        model_path: Path to the trained Ludwig model.
        allowed_origins: CORS allowed origins.
        prediction_timeout: Maximum seconds to wait for a prediction before
            returning HTTP 504.

    Returns:
        FastAPI application.
    """
    app = FastAPI(
        title="Ludwig Inference Server",
        description="Production-ready model serving with auto-generated schemas, "
        "Prometheus metrics, structured logging, and timeout handling.",
        version="0.12.0",
    )

    if allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])

    if model_path:
        model_manager.load(model_path)

    # Build typed schemas after (potential) model load so they reflect actual features
    PredictRequest = build_request_schema(model_manager.config or {})
    PredictResponse = build_response_schema(model_manager.config or {})

    # ------------------------------------------------------------------ #
    # Middleware: structured request/response logging                      #
    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log method, path, status code, and duration for every HTTP request."""
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        logger.info(
            "http_request method=%s path=%s status=%d duration_seconds=%.4f client=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            request.client.host if request.client else "unknown",
        )
        return response

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #
    @app.get("/")
    def health():
        return {"status": "healthy", "model_loaded": model_manager.model is not None}

    # ------------------------------------------------------------------ #
    # Model info                                                           #
    # ------------------------------------------------------------------ #
    @app.get("/info")
    def model_info():
        if model_manager.config is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "model_type": model_manager.config.get("model_type", "ecd"),
            "input_features": [
                {"name": n, "type": f.get("type")}
                for n, f in zip(model_manager._input_feature_names, model_manager.config.get("input_features", []))
            ],
            "output_features": [
                {"name": n, "type": f.get("type")}
                for n, f in zip(model_manager._output_feature_names, model_manager.config.get("output_features", []))
            ],
        }

    # ------------------------------------------------------------------ #
    # Prometheus metrics                                                   #
    # ------------------------------------------------------------------ #
    if METRICS_AVAILABLE:

        @app.get("/metrics")
        def metrics():
            from starlette.responses import Response

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ #
    # Single-row prediction                                                #
    # ------------------------------------------------------------------ #
    @app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest, model=Depends(get_model)):  # type: ignore[valid-type]
        start = time.monotonic()

        # Convert Pydantic model to dict, drop unset/None fields
        entry = {k: v for k, v in body.model_dump().items() if v is not None}

        # Validate all required input features are present
        missing = model_manager._input_features_set - set(entry.keys())
        if missing:
            _record_error("predict")
            raise HTTPException(
                status_code=400,
                detail=f"Missing input features: {sorted(missing)}",
            )

        _log_request("predict", sorted(entry.keys()), batch_size=1)

        try:
            resp_df, _ = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.predict(dataset=[entry], data_format=dict),
                ),
                timeout=prediction_timeout,
            )
        except TimeoutError:
            _record_error("predict")
            logger.error("prediction_timeout endpoint=predict timeout_seconds=%.1f", prediction_timeout)
            raise HTTPException(status_code=504, detail=f"Prediction timed out after {prediction_timeout}s")
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            _record_error("predict")
            raise HTTPException(status_code=500, detail=str(exc))

        result = _numpy_safe(resp_df.to_dict("records")[0])
        latency = time.monotonic() - start
        _log_response("predict", model_manager._output_feature_names, latency)
        _record_success("predict", latency)
        return JSONResponse(result)

    # ------------------------------------------------------------------ #
    # Batch prediction                                                     #
    # ------------------------------------------------------------------ #
    @app.post("/batch_predict")
    async def batch_predict(request: Request, model=Depends(get_model)):
        start = time.monotonic()
        try:
            body = await request.json()
            df = pd.DataFrame(body if isinstance(body, list) else [body])
        except Exception as exc:
            _record_error("batch_predict")
            raise HTTPException(status_code=400, detail=f"Invalid request body: {exc}")

        missing = model_manager._input_features_set - set(df.columns)
        if missing:
            _record_error("batch_predict")
            raise HTTPException(
                status_code=400,
                detail=f"Missing input features: {sorted(missing)}",
            )

        _log_request("batch_predict", sorted(df.columns.tolist()), batch_size=len(df))

        try:
            resp_df, _ = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.predict(dataset=df),
                ),
                timeout=prediction_timeout,
            )
        except TimeoutError:
            _record_error("batch_predict")
            logger.error("prediction_timeout endpoint=batch_predict timeout_seconds=%.1f", prediction_timeout)
            raise HTTPException(status_code=504, detail=f"Prediction timed out after {prediction_timeout}s")
        except Exception as exc:
            logger.exception("Batch prediction failed: %s", exc)
            _record_error("batch_predict")
            raise HTTPException(status_code=500, detail=str(exc))

        result = _numpy_safe(resp_df.to_dict("split"))
        latency = time.monotonic() - start
        _log_response("batch_predict", model_manager._output_feature_names, latency)
        _record_success("batch_predict", latency)
        return JSONResponse(result)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    allowed_origins: list[str] | None = None,
    prediction_timeout: float = 30.0,
) -> None:
    """Run the Ludwig serving application."""
    import uvicorn

    app = create_app(
        model_path=model_path,
        allowed_origins=allowed_origins,
        prediction_timeout=prediction_timeout,
    )
    uvicorn.run(app, host=host, port=port)
