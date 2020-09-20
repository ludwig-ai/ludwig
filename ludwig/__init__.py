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
import logging
import sys

import absl.logging

from ludwig.globals import LUDWIG_VERSION as __version__

# Tensorflow 1.14 has compatibility issues with python native logging
# This was one of the suggested solutions to continue using native logging
# https://github.com/tensorflow/tensorflow/issues/26691
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

logger = logging.getLogger(__name__)
# Default logging level for the project
logger.setLevel(logging.INFO)

# Configure stream handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Set formatter
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)
