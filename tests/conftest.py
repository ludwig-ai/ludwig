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
import os
import tempfile
import uuid

import pytest


@pytest.fixture()
def csv_filename():
    """Yields a csv filename for holding temporary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, uuid.uuid4().hex[:10].upper() + ".csv")
        yield csv_filename


@pytest.fixture()
def yaml_filename():
    """Yields a yaml filename for holding a temporary config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_filename = os.path.join(tmpdir, "model_def_" + uuid.uuid4().hex[:10].upper() + ".yaml")
        yield yaml_filename
