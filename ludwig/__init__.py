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

from ludwig.globals import LUDWIG_VERSION as __version__  # noqa

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

# Disable annoying message about NUMEXPR_MAX_THREADS
logging.getLogger("numexpr").setLevel(logging.WARNING)
