import os
from random import randint, seed

import pytest

from ludwig.model_export.onnx_exporter import OnnxExporter

seed(1)


def test_convert_torch_to_onnx():
    onnx_exporter = OnnxExporter()
    input_model_name = "sample_model"
    model_id = randint(1000, 10000)
    output_model_name = input_model_name + ".onnx"
    export_path = f"modeloutputs/{model_id}/run/model"
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, export_path)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    model_path = f"./saved_models"
    onnx_exporter.export_classifier(model_id, model_path, export_path, input_model_name, output_model_name)
    onnx_exporter.check_model_export(model_id, final_directory, output_model_name)
    assert os.path.isfile(os.path.join(export_path, output_model_name))
