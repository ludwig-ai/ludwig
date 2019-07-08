#! /usr/bin/env python
# coding=utf-8
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

from ludwig.contrib import contrib_command
from ludwig.utils.print_utils import logging_level_registry
import json

from ludwig.api import LudwigModel


logger = logging.getLogger(__name__)


def start_server(
    model_path,
    logging_level,
    host,
    port
):

    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    global model
    model = LudwigModel.load(model_path)

    global input_features
    input_features = {
        f['name'] for f in model.model_definition['input_features']
    }

    @app.post('/predict')
    async def predict(request: Request):
        data_json = await request.body()
        entries = json.loads(data_json)

        for entry in entries:
            if (entry.keys() & input_features) != input_features:
                return JSONResponse({"error": "entries must contain all input features"},
                                    status_code=400)

        try:
            resp = model.predict(data_dict=entries).to_dict('records')
            return JSONResponse(resp)
        except Exception as e:
            logger.error("Error: {}".format(str(e)))
            return JSONResponse({"error": "Unexpected Error: could not run inference on model"},
                                status_code=500)

    uvicorn.run(app, host=host, port=port)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script serves a pretrained model',
        prog='ludwig serve',
        usage='%(prog)s [options]'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument(
        '-m',
        '--model_path',
        help='model to load',
        required=True
    )

    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    # ----------------
    # Server parameters
    # ----------------
    parser.add_argument(
        '-p',
        '--port',
        help='port for server',
        default=8000,
        type=int,
    )

    parser.add_argument(
        '-H',
        '--host',
        help='host for server',
        default='0.0.0.0'
    )

    args = parser.parse_args(sys_argv)

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )

    start_server(**vars(args))


if __name__ == '__main__':
    contrib_command("serve", *sys.argv)
    cli(sys.argv[1:])
