import unittest
from unittest.mock import MagicMock, patch

import onnx

from ludwig.api import LudwigModel
from ludwig.model_export.base_model_exporter import LudwigTorchWrapper
from ludwig.model_export.onnx_exporter import OnnxExporter


class TestOnnxExporter(unittest.TestCase):
    @patch.object(LudwigModel, "load")
    @patch.object(LudwigTorchWrapper, "eval")
    @patch("torch.onnx")
    def test_onnx_export(
        self,
        mock_onnx,
        mock_ludwig_torch_wrapper_eval,
        mock_ludwig_model_load,
    ):
        sample_model_path = MagicMock()
        sample_export_path = MagicMock()
        sample_output_model_name = MagicMock()
        mock_ludwig_model_load.return_value = MagicMock()
        mock_onnx.export.return_value = MagicMock()
        onnx_exporter = OnnxExporter()

        onnx_exporter.export(sample_model_path, sample_export_path, sample_output_model_name, False)

        mock_ludwig_torch_wrapper_eval.assert_called_once()
        mock_ludwig_model_load.assert_called_once()

    @patch("torch.quantization.quantize_dynamic")
    def test_ludwig_quantization(self, mock_quantize_dynamic):
        quantized_model = MagicMock(spec=LudwigModel)
        mock_quantize_dynamic.return_value = quantized_model
        mock_ludwig_model = MagicMock(spec=LudwigModel)
        with patch.object(LudwigModel, "load", return_value=mock_ludwig_model):
            onnx_exporter = OnnxExporter()
            quantized_model = onnx_exporter.quantize_ludwig_model(mock_ludwig_model)
        self.assertIsInstance(quantized_model, LudwigModel)

    def test_quantize_onnx_model(self):
        mock_model_path = "tests/ludwig/model_export/sampleonnxmodel/preprocessed.onnx"
        mock_output_path = "tests/ludwig/model_export/sampleonnxmodel/quantized_squeeze.onnx"

        onnx_exporter = OnnxExporter()
        onnx_exporter.check_model_export(mock_model_path)
        onnx_exporter.quantize_onnx_model(mock_model_path, mock_output_path)
        onnx_model = onnx.load(mock_output_path)
        self.assertIsInstance(onnx_model, onnx.ModelProto)

        # Current Issues with this specific unittest:
        # ..Quantization parameters for tensor:"x" not specified
        # Quantization parameters for tensor:"/features/features.2/MaxPool_output_0" not specified
        # Quantization parameters for tensor:"/features/features.3/squeeze_activation/Relu_output_0" not specified
        # Quantization parameters for tensor:"/features/features.3/Concat_output_0" not specified
        # ..
        # Tests pass but not without these warnings.
