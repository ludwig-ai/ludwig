from abc import ABC, abstractmethod

import torch
import onnx
from ludwig.api import LudwigModel
import os
from ludwig.model_export.base_model_exporter import LudwigTorchWrapper


class OnnxExporter(ABC):

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