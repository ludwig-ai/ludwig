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
import os
import sys
import tempfile

from ludwig.api import LudwigModel
from ludwig.contrib import contrib_command
from ludwig.utils.print_utils import logging_level_registry

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import FastAPI
    from starlette.datastructures import UploadFile
    from starlette.requests import Request
    from starlette.responses import JSONResponse
except ImportError:
    logger.error(
        ' fastapi and other serving dependencies are not installed. '
        'In order to install all serving dependencies run '
        'pip install ludwig[serve]'
    )
    sys.exit(-1)

ALL_FEATURES_PRESENT_ERROR = {"error": "entry must contain all input features"}

COULD_NOT_RUN_INFERENCE_ERROR = {
    "error": "Unexpected Error: could not run inference on model"}


def server(model):
    app = FastAPI()

    input_features = {
        f['name'] for f in model.model_definition['input_features']
    }

    @app.get('/')
    def check_health():
        return JSONResponse({"message": "Ludwig server is up"})

    @app.post('/predict')
    async def predict(request: Request):
        form = await request.form()
        files, entry = convert_input(form)

        try:
            if (entry.keys() & input_features) != input_features:
                return JSONResponse(ALL_FEATURES_PRESENT_ERROR,
                                    status_code=400)
            try:
                resp = model.predict(data_dict=[entry]).to_dict('records')[0]
                return JSONResponse(resp)
            except Exception as e:
                logger.error("Error: {}".format(str(e)))
                return JSONResponse(COULD_NOT_RUN_INFERENCE_ERROR,
                                    status_code=500)
        finally:
            for f in files:
                os.remove(f.name)

    return app


def convert_input(form):
    '''
    Returns a new input and a list of files to be cleaned up
    '''
    new_input = {}
    files = []
    for k, v in form.multi_items():
        if type(v) == UploadFile:
            # Convert UploadFile to a NamedTemporaryFile to ensure it's on the disk
            suffix = os.path.splitext(v.filename)[1]
            named_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix)
            files.append(named_file)
            named_file.write(v.file.read())
            named_file.close()
            new_input[k] = named_file.name
        else:
            new_input[k] = v

    return (files, new_input)


def run_server(model_path, host, port):
    model = LudwigModel.load(model_path)
    app = server(model)
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
        help='port for server (default: 8000)',
        default=8000,
        type=int,
    )

    parser.add_argument(
        '-H',
        '--host',
        help='host for server (default: 0.0.0.0)',
        default='0.0.0.0'
    )

    args = parser.parse_args(sys_argv)

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )

    run_server(args.model_path, args.host, args.port)


if __name__ == '__main__':
    contrib_command("serve", *sys.argv)
    cli(sys.argv[1:])
