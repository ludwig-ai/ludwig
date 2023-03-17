#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
import io
import json
import logging
import os
import sys
import tempfile

import pandas as pd
import torch
from torchvision.io import decode_image

from ludwig.api import LudwigModel
from ludwig.constants import AUDIO, COLUMN
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig
from ludwig.utils.server_utils import NumpyJSONResponse

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import FastAPI
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

ALL_FEATURES_PRESENT_ERROR = {"error": "entry must contain all input features"}

COULD_NOT_RUN_INFERENCE_ERROR = {"error": "Unexpected Error: could not run inference on model"}


def server(model, allowed_origins=None):
    middleware = [Middleware(CORSMiddleware, allow_origins=allowed_origins)] if allowed_origins else None
    app = FastAPI(middleware=middleware)

    config = model.config
    input_features = {f[COLUMN] for f in config["input_features"]}

    @app.get("/")
    def check_health():
        return NumpyJSONResponse({"message": "Ludwig server is up"})

    @app.post("/predict")
    async def predict(request: Request):
        try:
            form = await request.form()
            entry, files = convert_input(form, model.model.input_features)
        except Exception:
            logger.exception("Failed to parse predict form")
            return NumpyJSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)

        try:
            if (entry.keys() & input_features) != input_features:
                missing_features = set(input_features) - set(entry.keys())
                return NumpyJSONResponse(
                    {
                        "error": "Data received does not contain all input features. "
                        f"Missing features: {missing_features}."
                    },
                    status_code=400,
                )
            try:
                resp, _ = model.predict(dataset=[entry], data_format=dict)
                resp = resp.to_dict("records")[0]
                return NumpyJSONResponse(resp)
            except Exception as exc:
                logger.exception(f"Failed to run predict: {exc}")
                return NumpyJSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)
        finally:
            for f in files:
                os.remove(f.name)

    @app.post("/batch_predict")
    async def batch_predict(request: Request):
        try:
            form = await request.form()
            data, files = convert_batch_input(form, model.model.input_features)
            data_df = pd.DataFrame.from_records(data["data"], index=data.get("index"), columns=data["columns"])
        except Exception:
            logger.exception("Failed to parse batch_predict form")
            return NumpyJSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)

        if (set(data_df.columns) & input_features) != input_features:
            missing_features = set(input_features) - set(data_df.columns)
            return NumpyJSONResponse(
                {
                    "error": "Data received does not contain all input features. "
                    f"Missing features: {missing_features}."
                },
                status_code=400,
            )
        try:
            resp, _ = model.predict(dataset=data_df)
            resp = resp.to_dict("split")
            return NumpyJSONResponse(resp)
        except Exception:
            logger.exception("Failed to run batch_predict: {}")
            return NumpyJSONResponse(COULD_NOT_RUN_INFERENCE_ERROR, status_code=500)

    return app


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
        if type(v) == UploadFile:
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
        if type(v) == UploadFile:
            file_index[v.filename] = v

    data = json.loads(form["dataset"])
    for row in data["data"]:
        for i in range(len(row)):
            if row[i] in file_index:
                feature_name = data["columns"][i]
                if input_features.get(feature_name).type() == AUDIO:
                    row[i] = _write_file(file_index[row[i]], files)
                else:
                    row[i] = _read_image_buffer(file_index[row[i]])

    return data, files


def run_server(
    model_path: str,
    host: str,
    port: int,
    allowed_origins: list,
) -> None:
    """Loads a pre-trained model and serve it on an http server.

    # Inputs

    :param model_path: (str) filepath to pre-trained model.
    :param host: (str, default: `0.0.0.0`) host ip address for the server to use.
    :param port: (int, default: `8000`) port number for the server to use.
    :param allowed_origins: (list) list of origins allowed to make cross-origin requests.

    # Return

    :return: (`None`)
    """
    # Use local backend for serving to use pandas DataFrames.
    model = LudwigModel.load(model_path, backend="local")
    app = server(model, allowed_origins)
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

    run_server(args.model_path, args.host, args.port, args.allowed_origins)


if __name__ == "__main__":
    cli(sys.argv[1:])
