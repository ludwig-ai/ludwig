"""Model introspection utilities.

Extracted from LudwigModel to reduce the god object. Provides:
- Weight collection
- Activation collection
- Schema generation
- Model summary
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ModelInspector:
    """Inspect and analyze a trained Ludwig model."""

    def __init__(self, model: torch.nn.Module, config: dict, training_set_metadata: dict):
        self.model = model
        self.config = config
        self.training_set_metadata = training_set_metadata

    def collect_weights(self, tensor_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Collect model weight tensors.

        Args:
            tensor_names: Specific parameter names to collect. None for all.

        Returns:
            List of dicts with 'name', 'shape', 'dtype', 'values' keys.
        """
        results = []
        for name, param in self.model.named_parameters():
            if tensor_names is None or name in tensor_names:
                results.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "requires_grad": param.requires_grad,
                        "num_elements": param.numel(),
                    }
                )
        return results

    def model_summary(self) -> dict[str, Any]:
        """Generate a summary of the model architecture.

        Returns:
            Dict with parameter counts, layer info, and feature details.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Model size in MB
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)

        # Count layers by type
        layer_counts = {}
        for module in self.model.modules():
            class_name = type(module).__name__
            layer_counts[class_name] = layer_counts.get(class_name, 0) + 1

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "model_size_mb": round(model_size_mb, 2),
            "layer_counts": layer_counts,
            "model_type": self.config.get("model_type", "ecd"),
            "combiner_type": self.config.get("combiner", {}).get("type", "concat"),
            "num_input_features": len(self.config.get("input_features", [])),
            "num_output_features": len(self.config.get("output_features", [])),
        }

    def feature_importance_proxy(self) -> dict[str, float]:
        """Estimate feature importance from encoder weight magnitudes.

        This is a rough proxy, not a rigorous importance measure. For proper
        feature importance, use SHAP or Captum via Ludwig's explain module.

        Returns:
            Dict mapping feature names to relative importance scores.
        """
        importance = {}
        if hasattr(self.model, "input_features"):
            for name, feature in self.model.input_features.items():
                total_weight_magnitude = 0.0
                for param in feature.parameters():
                    total_weight_magnitude += param.abs().mean().item()
                importance[name] = total_weight_magnitude

        # Normalize to 0-1 range
        if importance:
            max_imp = max(importance.values())
            if max_imp > 0:
                importance = {k: v / max_imp for k, v in importance.items()}

        return importance
