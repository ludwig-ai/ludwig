# MetaFormer integration package init
from .metaformer_models import get_registered_model, default_cfgs  # noqa: F401
from .metaformer_stacked_cnn import (
    MetaFormerStackedCNN,
    CAFormerStackedCNN,
    patch_ludwig_comprehensive,
    patch_ludwig_direct,
    patch_ludwig_robust,
    list_metaformer_models,
    get_metaformer_backbone_names,
    metaformer_model_exists,
    describe_metaformer_model,
)  # noqa: F401

def patch():
    """Convenience one-call patch entrypoint (alias of patch_ludwig_comprehensive)."""
    return patch_ludwig_comprehensive()

__all__ = [
    "MetaFormerStackedCNN",
    "CAFormerStackedCNN",
    "patch_ludwig_comprehensive",
    "patch_ludwig_direct",
    "patch_ludwig_robust",
    "list_metaformer_models",
    "get_metaformer_backbone_names",
    "metaformer_model_exists",
    "describe_metaformer_model",
    "get_registered_model",
    "default_cfgs",
    "patch",
]
