import torch
from ludwig.api import LudwigModel
from abc import ABC, abstractmethod

class OnnxExporter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def forward(self, x):
        return self.model({"image_path": x})

    def export_classifier(self, model_path, export_path):
        ludwig_model = LudwigModel.load(model_path)
        model = OnnxExporter(ludwig_model.model)  # Wrap the model

        width = ludwig_model.config["input_features"][0]["preprocessing"]["width"]
        height = ludwig_model.config["input_features"][0]["preprocessing"]["height"]
        example_input = torch.randn(1, 3, width, height, requires_grad=True)

        torch.onnx.export(
            model,
            example_input,
            export_path,
            opset_version=18,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["combiner_hidden_1", "output", "combiner_hidden_2"],
        )


    @abstractmethod
    def quantize(self, path_fp32, path_int8):
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(path_fp32, path_int8)  # type: ignore