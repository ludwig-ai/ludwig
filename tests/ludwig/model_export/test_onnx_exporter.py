import unittest
from unittest.mock import MagicMock, patch

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

        onnx_exporter.export(sample_model_path, sample_export_path, sample_output_model_name)

        mock_ludwig_torch_wrapper_eval.assert_called_once()
        mock_ludwig_model_load.assert_called_once()
