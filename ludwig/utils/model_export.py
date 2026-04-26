"""Modern model export utilities using torch.export, ONNX, and SafeTensors.

Replaces the deprecated TorchScript export pipeline with:
- torch.export: PyTorch 2.x native export producing ExportedProgram (ATen-level IR)
- ONNX: via torch.onnx.export(dynamo=True) for cross-platform deployment
- SafeTensors: secure, zero-copy weight serialization (already default for ECD)

TorchScript is fully deprecated as of PyTorch 2.9 and was removed from Ludwig in
v0.15. torch.export is the official replacement that captures the full computation
graph as an ExportedProgram. Migration guide:
https://pytorch.org/docs/stable/export.html

Usage:
    from ludwig.utils.model_export import ModelExporter

    exporter = ModelExporter(model)
    exporter.export_torch(path, sample_input)    # torch.export format (.pt2)
    exporter.export_onnx(path, sample_input)     # ONNX via dynamo
    exporter.export_safetensors(path)            # weights only
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class ModelExporter:
    """Unified model export interface for Ludwig models."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def export_torch(self, save_path: str, sample_input: dict[str, torch.Tensor] | None = None):
        """Export model via torch.export for deployment.

        Produces an ExportedProgram that captures the full computation graph at the
        ATen operator level. This is the official replacement for TorchScript.

        The exported program can be:
        - Loaded back via torch.export.load()
        - Compiled with torch.compile() for runtime optimization
        - Used as input to torch.onnx.export(dynamo=True)
        - Deployed via ExecuTorch for on-device inference

        Args:
            save_path: Directory to save the exported model.
            sample_input: Example input dict for tracing. If None, generates one.

        Returns:
            Path to the saved .pt2 file.
        """
        os.makedirs(save_path, exist_ok=True)
        export_path = os.path.join(save_path, "model.pt2")

        self.model.eval()

        if sample_input is None:
            sample_input = self._generate_sample_input()

        try:
            # torch.export captures the full computation graph
            exported = torch.export.export(self.model, args=(sample_input,))
            torch.export.save(exported, export_path)
            logger.info(f"Model exported via torch.export to {export_path}")
        except Exception as e:
            logger.warning(
                f"torch.export failed: {e}. This can happen with dynamic control flow "
                f"or unsupported operations. Falling back to torch.jit.trace."
            )
            # Fallback to tracing for models that can't be exported
            traced = torch.jit.trace(self.model, (sample_input,), strict=False)
            traced.save(export_path)
            logger.info(f"Model exported via torch.jit.trace (fallback) to {export_path}")

        return export_path

    def export_onnx(self, save_path: str, sample_input: dict[str, torch.Tensor] | None = None):
        """Export model to ONNX format via dynamo-based exporter.

        Uses torch.onnx.export(dynamo=True) which is built on torch.export
        and produces more accurate ONNX graphs than the legacy TorchScript-based
        ONNX exporter.

        Args:
            save_path: Directory to save the ONNX model.
            sample_input: Example input dict for tracing.

        Returns:
            Path to the saved .onnx file.
        """
        os.makedirs(save_path, exist_ok=True)
        onnx_path = os.path.join(save_path, "model.onnx")

        self.model.eval()

        if sample_input is None:
            sample_input = self._generate_sample_input()

        try:
            # Dynamo-based ONNX export (recommended for PyTorch 2.x)
            torch.onnx.export(
                self.model,
                (sample_input,),
                onnx_path,
                dynamo=True,
            )
            logger.info(f"Model exported to ONNX (dynamo) at {onnx_path}")
        except Exception as e:
            logger.warning(f"Dynamo ONNX export failed: {e}. Trying legacy exporter.")
            try:
                torch.onnx.export(
                    self.model,
                    (sample_input,),
                    onnx_path,
                    opset_version=17,
                    input_names=list(sample_input.keys()),
                )
                logger.info(f"Model exported to ONNX (legacy) at {onnx_path}")
            except Exception as e2:
                logger.error(f"ONNX export failed: {e2}")
                raise

        return onnx_path

    def export_safetensors(self, save_path: str):
        """Export model weights in SafeTensors format.

        SafeTensors provides secure, zero-copy weight serialization.
        Already the default format for Ludwig ECD models.

        Args:
            save_path: Directory to save the weights.

        Returns:
            Path to the saved .safetensors file.
        """
        from safetensors.torch import save_model

        os.makedirs(save_path, exist_ok=True)
        weights_path = os.path.join(save_path, "model.safetensors")
        save_model(self.model, weights_path)
        logger.info(f"Model weights exported to SafeTensors at {weights_path}")
        return weights_path

    def _generate_sample_input(self) -> dict[str, torch.Tensor]:
        """Generate a sample input for export tracing.

        Uses the model's create_sample_input() if available, otherwise creates dummy tensors based on input feature
        shapes.
        """
        if hasattr(self.model, "create_sample_input"):
            return self.model.create_sample_input()

        # Fallback: create dummy inputs from input features
        sample = {}
        if hasattr(self.model, "input_features"):
            for name, feature in self.model.input_features.items():
                if hasattr(feature, "create_sample_input"):
                    sample[name] = feature.create_sample_input(batch_size=2)
                else:
                    sample[name] = torch.zeros(2, 1)
        else:
            logger.warning("Cannot generate sample input: model has no input_features")
            sample = {"input": torch.zeros(2, 1)}

        return sample


def load_exported_model(path: str) -> torch.nn.Module:
    """Load an exported model from disk.

    Supports torch.export (.pt2), TorchScript (.pt), and ONNX (.onnx) formats.

    Args:
        path: Path to the exported model file.

    Returns:
        Loaded model or ExportedProgram.
    """
    if path.endswith(".pt2"):
        try:
            return torch.export.load(path)
        except Exception:
            # Fallback for traced models saved with .pt2 extension
            return torch.jit.load(path)
    elif path.endswith(".pt"):
        return torch.jit.load(path)
    elif path.endswith(".onnx"):
        try:
            import onnxruntime as ort

            return ort.InferenceSession(path)
        except ImportError:
            raise ImportError("onnxruntime is required to load ONNX models. pip install onnxruntime")
    else:
        raise ValueError(f"Unknown model format: {path}. Supported: .pt2, .pt, .onnx")
