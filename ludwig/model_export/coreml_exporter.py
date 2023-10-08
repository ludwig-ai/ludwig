import torch
from ludwig.api import LudwigModel
from abc import ABC, abstractmethod
from ludwig.model_export.base_model_exporter import BaseModelExporter
import torch
import coremltools as ct
from model import MyNet

 class CoreMLExporter(BaseModelExporter):

    def export_classifier(self, model_path, export_path):
        ludwig_model = LudwigModel.load(model_path)
        model = CoreMLExporter(ludwig_model.model)  # Wrap the model
        #model.eval()  # inference mode, is this needed.. I think onnx export does this for us

        width = ludwig_model.config["input_features"][0]["preprocessing"]["width"]
        height = ludwig_model.config["input_features"][0]["preprocessing"]["height"]
        example_input = torch.randn(1, 3, width, height, requires_grad=True)

        # Create the model and load the weights
        model = MyNet()
        model.load_state_dict(torch.load(model_path))
        # Create dummy input
        dummy_input = torch.rand(1, 3, 32, 32)
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        # Create the input image type
        input_image = ct.ImageType(name=example_input, shape=(1, 3, 32, 32), scale=1/255)
        # Convert the model
        coreml_model = ct.convert(traced_model, inputs=[input_image])

        # Modify the output's name to "my_output" in the spec
        spec = coreml_model.get_spec()
        ct.utils.rename_feature(spec, "81", export_path)

        # Re-create the model from the updated spec
        coreml_model_updated = ct.models.MLModel(spec)

        # Save the CoreML model
        coreml_model_updated.save(export_path)