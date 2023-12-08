import os

import coremltools as ct
import torch

from ludwig.api import LudwigModel
from ludwig.model_export.base_model_exporter import BaseModelExporter, LudwigTorchWrapper


class CoreMLExporter(BaseModelExporter):
    """Class that abstracts the convertion of torch to onnx."""

    def export(self, model_path, export_path, output_model_name):
        ludwig_model = LudwigModel.load(model_path)
        model = ludwig_model.model.to("cpu").eval()

        # option 1, works but hacky
        # encoder = model.input_features.module_dict.image_path__ludwig.encoder_obj.model  # type: ignore
        # decoder = model.output_features.module_dict.label__ludwig.decoder_obj  # type: ignore
        # model = torch.nn.Sequential(encoder, decoder)

        # option 2, doesn't work
        # throws error: RuntimeError: PyTorch convert function for op 'dictconstruct' not implemented.
        model = LudwigTorchWrapper(model)

        width = ludwig_model.config["input_features"][0]["preprocessing"]["width"]
        height = ludwig_model.config["input_features"][0]["preprocessing"]["height"]
        example_input = torch.randn(1, 3, width, height)

        traced_model = torch.jit.trace(model, example_input, strict=False)

        image_input = ct.ImageType(
            name="image",
            shape=example_input.shape,
            scale=1 / 255.0,
            bias=[0.0, 0.0, 0.0],
        )

        print(f"converting to core_ml, input_input={image_input}")

        coreml_model = ct.convert(
            traced_model,
            convert_to="neuralnetwork",
            inputs=[image_input],
            debug=True,
        )

        coreml_path = os.path.join(export_path, output_model_name)
        coreml_model.save(coreml_path)  # type: ignore

    def check_model_export(self, path):
        coreml_model = ct.models.MLModel(path)
        coreml_model.get_spec()
