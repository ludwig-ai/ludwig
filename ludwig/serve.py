#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import asyncio
import io
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, create_model, Field
from torchvision.io import decode_image

from ludwig.api import LudwigModel
from ludwig.constants import AUDIO, COLUMN
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from starlette.datastructures import UploadFile
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
except ImportError as e:
    logger.error(e)
    logger.error(
        " fastapi and other serving dependencies cannot be loaded"
        "and may have not been installed. "
        "In order to install all serving dependencies run "
        "pip install ludwig[serve]"
    )
    sys.exit(-1)

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
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Error constants
# ---------------------------------------------------------------------------
ALL_FEATURES_PRESENT_ERROR = {"error": "entry must contain all input features"}
COULD_NOT_RUN_INFERENCE_ERROR = {"error": "Unexpected Error: could not run inference on model"}


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
    "image": str | bytes | list,
    "audio": str | bytes | list,
    "timeseries": list,
}


def _feature_python_type(ftype: str) -> type:
    return _FEATURE_TYPE_TO_PYTHON.get(ftype, Any)


def build_request_schema(config: dict) -> type[BaseModel]:
    """Dynamically create a Pydantic request model from Ludwig input feature configs.

    Each input feature becomes an optional field (None default) so that missing features can be detected and reported
    with a clear error message rather than a Pydantic validation error.
    """
    fields: dict[str, Any] = {}
    for feat in config.get("input_features", []):
        name = feat.get("name") or feat.get(COLUMN) or feat.get("column")
        ftype = feat.get("type", "")
        py_type = _feature_python_type(ftype)
        # Make every field Optional so callers get a descriptive error, not a 422
        fields[name] = (py_type | None, Field(default=None, description=f"Input feature '{name}' of type '{ftype}'"))

    return create_model("PredictRequest", **fields)


def build_response_schema(config: dict) -> type[BaseModel]:
    """Dynamically create a Pydantic response model from Ludwig output feature configs.

    The exact column names produced by Ludwig's post-processor (e.g. ``age_predictions``, ``churn_probabilities``) are
    declared as optional fields so extra columns don't break validation.
    """
    fields: dict[str, Any] = {}
    for feat in config.get("output_features", []):
        name = feat.get("name") or feat.get(COLUMN) or feat.get("column")
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
            # Catch-all: accept anything
            fields[f"{name}_predictions"] = (Any, None)

    return create_model("PredictResponse", **fields)


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------
def _record_success(endpoint: str, latency: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        _REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        _REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


def _record_error(endpoint: str) -> None:
    if _PROMETHEUS_AVAILABLE:
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
# Server factory
# ---------------------------------------------------------------------------
def server(
    model: "LudwigModel",
    allowed_origins: list[str] | None = None,
    prediction_timeout: float = 30.0,
) -> "FastAPI":
    """Build a FastAPI application for serving a Ludwig model.

    Args:
        model: A loaded :class:`~ludwig.api.LudwigModel` instance.
        allowed_origins: List of CORS-allowed origins.  ``None`` disables CORS.
        prediction_timeout: Seconds to wait for a prediction before returning
            HTTP 504 Gateway Timeout.

    Returns:
        A :class:`fastapi.FastAPI` application ready to be served with uvicorn.
    """
    middleware = [Middleware(CORSMiddleware, allow_origins=allowed_origins)] if allowed_origins else None
    app = FastAPI(
        title="Ludwig Inference Server",
        description="Auto-generated schemas, Prometheus metrics, structured logging, and timeout handling.",
        middleware=middleware,
    )

    config = model.config
    input_features = {f[COLUMN] for f in config["input_features"]}
    output_feature_names = [f.get("name", f.get(COLUMN, "")) for f in config.get("output_features", [])]

    # Build typed Pydantic schemas for OpenAPI docs.
    # PredictResponse is wired into /predict as a response_model.
    # The request schema is attached via openapi_extra since the actual transport
    # is multipart/form-data rather than a JSON body.
    predict_request_schema = build_request_schema(config).model_json_schema()
    PredictResponse = build_response_schema(config)

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #
    @app.get("/")
    def check_health():
        return JSONResponse({"message": "Ludwig server is up"})

    # ------------------------------------------------------------------ #
    # Prometheus metrics                                                   #
    # ------------------------------------------------------------------ #
    if _PROMETHEUS_AVAILABLE:

        @app.get("/metrics")
        def metrics():
            from starlette.responses import Response

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ #
    # Single-row prediction                                                #
    # ------------------------------------------------------------------ #
    @app.post(
        "/predict",
        response_model=PredictResponse,
        openapi_extra={"requestBody": {"content": {"multipart/form-data": {"schema": predict_request_schema}}}},
    )
    async def predict(request: Request):
        start = time.monotonic()
        files: list = []
        try:
            form = await request.form()
            entry, files = convert_input(form, model.model.input_features)
        except Exception:
            logger.exception("Failed to parse predict form")
            _record_error("predict")
            return JSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)

        try:
            if (entry.keys() & input_features) != input_features:
                missing_features = input_features - set(entry.keys())
                _record_error("predict")
                return JSONResponse(
                    {
                        "error": "Data received does not contain all input features. "
                        f"Missing features: {missing_features}."
                    },
                    status_code=400,
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
                logger.error(
                    "prediction_timeout endpoint=predict timeout_seconds=%.1f",
                    prediction_timeout,
                )
                return JSONResponse(
                    {"error": f"Prediction timed out after {prediction_timeout}s"},
                    status_code=504,
                )

            resp = _numpy_safe(resp_df.to_dict("records")[0])
            latency = time.monotonic() - start
            _log_response("predict", output_feature_names, latency)
            _record_success("predict", latency)
            return JSONResponse(resp)

        except Exception as exc:
            logger.exception("Failed to run predict: %s", exc)
            _record_error("predict")
            return JSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)
        finally:
            for f in files:
                os.remove(f.name)

    # ------------------------------------------------------------------ #
    # Batch prediction                                                     #
    # ------------------------------------------------------------------ #
    @app.post("/batch_predict")
    async def batch_predict(request: Request):
        start = time.monotonic()
        files: list = []
        try:
            form = await request.form()
            data, files = convert_batch_input(form, model.model.input_features)
            data_df = pd.DataFrame.from_records(
                data["data"],
                index=data.get("index"),
                columns=data["columns"],
            )
        except Exception:
            logger.exception("Failed to parse batch_predict form")
            _record_error("batch_predict")
            return JSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)

        if (set(data_df.columns) & input_features) != input_features:
            missing_features = input_features - set(data_df.columns)
            _record_error("batch_predict")
            return JSONResponse(
                {
                    "error": "Data received does not contain all input features. "
                    f"Missing features: {missing_features}."
                },
                status_code=400,
            )

        _log_request("batch_predict", sorted(data_df.columns.tolist()), batch_size=len(data_df))

        try:
            try:
                resp_df, _ = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: model.predict(dataset=data_df),
                    ),
                    timeout=prediction_timeout,
                )
            except TimeoutError:
                _record_error("batch_predict")
                logger.error(
                    "prediction_timeout endpoint=batch_predict timeout_seconds=%.1f",
                    prediction_timeout,
                )
                return JSONResponse(
                    {"error": f"Prediction timed out after {prediction_timeout}s"},
                    status_code=504,
                )

            resp = _numpy_safe(resp_df.to_dict("split"))
            latency = time.monotonic() - start
            _log_response("batch_predict", output_feature_names, latency)
            _record_success("batch_predict", latency)
            return JSONResponse(resp)

        except Exception:
            logger.exception("Failed to run batch_predict")
            _record_error("batch_predict")
            return JSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)
        finally:
            for f in files:
                os.remove(f.name)

    return app


# ---------------------------------------------------------------------------
# Form / file helpers (unchanged from original)
# ---------------------------------------------------------------------------
def _write_file(v, files):
    # Convert UploadFile to a NamedTemporaryFile to ensure it's on the disk
    suffix = os.path.splitext(v.filename)[1]
    named_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    files.append(named_file)
    named_file.write(v.file.read())
    named_file.close()
    return named_file.name


def _read_image_buffer(v):
    # read bytes sent via REST API and convert to image tensor
    # in [channels, height, width] format
    byte_string = io.BytesIO(v.file.read()).read()
    image = decode_image(torch.frombuffer(byte_string, dtype=torch.uint8))
    return image  # channels, height, width


def convert_input(form, input_features):
    """Returns a new input and a list of files to be cleaned up."""
    new_input = {}
    files = []
    for k, v in form.multi_items():
        if isinstance(v, UploadFile):
            # check if audio or image file
            if input_features.get(k).type() == AUDIO:
                new_input[k] = _write_file(v, files)
            else:
                new_input[k] = _read_image_buffer(v)
        else:
            new_input[k] = v

    return new_input, files


def convert_batch_input(form, input_features):
    """Returns a new input and a list of files to be cleaned up."""
    file_index = {}
    files = []
    for k, v in form.multi_items():
        if isinstance(v, UploadFile):
            file_index[v.filename] = v

    data = json.loads(form["dataset"])
    for row in data["data"]:
        for i, value in enumerate(row):
            if value in file_index:
                feature_name = data["columns"][i]
                if input_features.get(feature_name).type() == AUDIO:
                    row[i] = _write_file(file_index[value], files)
                else:
                    row[i] = _read_image_buffer(file_index[value])

    return data, files


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_server(
    model_path: str,
    host: str,
    port: int,
    allowed_origins: list,
    prediction_timeout: float = 30.0,
) -> None:
    """Load a pre-trained model and serve it on an HTTP server.

    Args:
        model_path: Filepath to pre-trained model.
        host: Host IP address for the server to use.
        port: Port number for the server to use.
        allowed_origins: List of origins allowed to make cross-origin requests.
        prediction_timeout: Seconds before a prediction request returns HTTP 504.
    """
    # Use local backend for serving to use pandas DataFrames.
    model = LudwigModel.load(model_path, backend="local")
    app = server(model, allowed_origins, prediction_timeout=prediction_timeout)
    uvicorn.run(app, host=host, port=port)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script serves a pretrained model", prog="ludwig serve", usage="%(prog)s [options]"
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument("-m", "--model_path", help="model to load", required=True)

    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    # ----------------
    # Server parameters
    # ----------------
    parser.add_argument(
        "-p",
        "--port",
        help="port for server (default: 8000)",
        default=8000,
        type=int,
    )

    parser.add_argument("-H", "--host", help="host for server (default: 0.0.0.0)", default="0.0.0.0")

    parser.add_argument(
        "-ao",
        "--allowed_origins",
        nargs="*",
        help="A list of origins that should be permitted to make cross-origin requests. "
        'Use "*" to allow any origin. See https://www.starlette.io/middleware/#corsmiddleware.',
    )

    parser.add_argument(
        "-t",
        "--prediction_timeout",
        help="Maximum seconds to wait for a prediction before returning HTTP 504 (default: 30.0)",
        default=30.0,
        type=float,
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("serve", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.serve")

    print_ludwig("Serve", LUDWIG_VERSION)

    run_server(args.model_path, args.host, args.port, args.allowed_origins, args.prediction_timeout)


if __name__ == "__main__":
    cli(sys.argv[1:])
