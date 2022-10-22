from dataclasses import dataclass

@dataclass
class InternalOutputFeatureConfig:
    """Base class for feature metadata."""

    column: str = None

    proc_column: str = None

    default_validation_metric: str = None

    input_size: int = None

    num_classes: int = None