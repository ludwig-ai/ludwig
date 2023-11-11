import os

import onnx
import torch

from ludwig.api import LudwigModel
from ludwig.model_export.base_model_exporter import BaseModelExporter, LudwigTorchWrapper


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
class OnnxExporter(BaseModelExporter):
    """Class that abstracts the convertion of torch to onnx."""

    def export(self, model_path, export_path, output_model_name):
        ludwig_model = LudwigModel.load(model_path)
        model = LudwigTorchWrapper(ludwig_model.model)  # Wrap the model
        model.eval()  # inference mode, is this needed.. I think onnx export does this for us

        width = ludwig_model.config["input_features"][0]["preprocessing"]["width"]
        height = ludwig_model.config["input_features"][0]["preprocessing"]["height"]
        example_input = torch.randn(1, 3, width, height, requires_grad=True)

        torch.onnx.export(
            model,
            example_input,
            os.path.join(export_path, output_model_name),
            opset_version=18,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["combiner_hidden_1", "output", "combiner_hidden_2"],
        )

    def check_model_export(self, path):
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
