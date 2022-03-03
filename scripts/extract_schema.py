# !/usr/bin/env python
# Copyright (c) 2021 Uber Technologies, Inc.
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

import json
import os
import subprocess
from pathlib import Path

# Import (unused) modules with marshmallow classes, necessary for generating list of subclasses.
import ludwig.combiners.combiners as lcc  # noqa: F401
import ludwig.models.trainer as lmt  # noqa: F401
import ludwig.modules.optimization_modules as lmo  # noqa: F401
import ludwig.utils.test_classes as lut  # noqa: F401
from ludwig.utils.marshmallow_schema_utils import BaseMarshmallowConfig, get_fully_qualified_class_name


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_mclass_paths():
    all_mclasses = list(all_subclasses(BaseMarshmallowConfig))
    all_mclasses += [BaseMarshmallowConfig, lcc.CommonTransformerConfig]
    return {cls.__name__: get_fully_qualified_class_name(cls) for cls in all_mclasses}


def get_pytkdocs_structure_for_path(path: str, docstring_style="restructured-text"):
    pytkdocs_subprocess = subprocess.Popen(
        ["pytkdocs"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    input_json = json.dumps({"objects": [{"path": path, "docstring_style": docstring_style}]})
    pytkdocs_output = pytkdocs_subprocess.communicate(input=input_json.encode())[0]
    return json.loads(pytkdocs_output.decode())


def extract_pytorch_structures():
    torch_structures = {}
    for opt in lmo.optimizer_registry:
        torch_type = lmo.optimizer_registry[opt][0]
        path = get_fully_qualified_class_name(torch_type)
        torch_structures[opt] = get_pytkdocs_structure_for_path(path, "google")
        parent_dir = str(Path(__file__).parent.parent)
        filename = (
            os.path.join(parent_dir, "ludwig/utils/documentation/torch_structures/", torch_type.__name__) + ".json"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            json.dump(get_pytkdocs_structure_for_path(path, "google"), outfile)
            outfile.write("\n")


def extract_marshmallow_structures():
    mclass_paths = get_mclass_paths()
    mclass_structures = {}
    for cls_name, path in mclass_paths.items():
        mclass_structures[cls_name] = get_pytkdocs_structure_for_path(path)
        parent_dir = str(Path(__file__).parent.parent)
        filename = os.path.join(parent_dir, "ludwig/utils/documentation/class_structures/", cls_name) + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            json.dump(get_pytkdocs_structure_for_path(path), outfile, indent=4, sort_keys=True, separators=(",", ": "))
            outfile.write("\n")


def main():
    extract_pytorch_structures()
    extract_marshmallow_structures()


if __name__ == "__main__":
    main()
