"""Typed metadata classes for Ludwig features and training sets.

Replaces the untyped TrainingSetMetadataDict = dict with structured dataclasses
that provide type safety, IDE autocomplete, and prevent key typo bugs.

These classes are backward-compatible: they can be constructed from dicts (via
from_dict class methods) and serialized back to dicts (via to_dict methods).
Existing code that accesses metadata as dicts continues to work during migration.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NumberMetadata:
    """Metadata for number features computed during preprocessing."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    q2: float | None = None
    q3: float | None = None
    ple_bin_edges: list[float] | None = None
    normalization: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "NumberMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class CategoryMetadata:
    """Metadata for category features computed during preprocessing."""

    idx2str: list[str] = field(default_factory=list)
    str2idx: dict[str, int] = field(default_factory=dict)
    str2freq: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    most_common_value: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "CategoryMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class TextMetadata:
    """Metadata for text features computed during preprocessing."""

    idx2str: list[str] = field(default_factory=list)
    str2idx: dict[str, int] = field(default_factory=dict)
    str2freq: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    max_sequence_length: int | None = None
    pad_idx: int = 0
    padding: str = "right"
    tokenizer_type: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "TextMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class BinaryMetadata:
    """Metadata for binary features computed during preprocessing."""

    str2bool: dict[str, bool] = field(default_factory=dict)
    bool2str: list[str] = field(default_factory=list)
    fallback_true_label: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "BinaryMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class ImageMetadata:
    """Metadata for image features computed during preprocessing."""

    num_channels: int = 3
    height: int = 0
    width: int = 0
    resize_method: str = "interpolate"
    infer_image_dimensions: bool = True
    infer_image_max_height: int = 256
    infer_image_max_width: int = 256
    infer_image_sample_size: int = 100
    scaling: str = "pixel_normalization"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ImageMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class SequenceMetadata:
    """Metadata for sequence features computed during preprocessing."""

    idx2str: list[str] = field(default_factory=list)
    str2idx: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    max_sequence_length: int | None = None
    pad_idx: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class AudioMetadata:
    """Metadata for audio features computed during preprocessing."""

    feature_dim: int = 0
    max_sequence_length: int | None = None
    sampling_rate: int = 16000

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "AudioMetadata":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# Registry mapping feature type strings to metadata classes
FEATURE_METADATA_CLASSES: dict[str, type] = {
    "number": NumberMetadata,
    "category": CategoryMetadata,
    "text": TextMetadata,
    "binary": BinaryMetadata,
    "image": ImageMetadata,
    "sequence": SequenceMetadata,
    "audio": AudioMetadata,
}


@dataclass
class TrainingSetMetadata:
    """Typed container for training set metadata.

    Replaces the untyped TrainingSetMetadataDict = dict with a structured container. Provides both typed access and
    dict-like backward compatibility.
    """

    features: dict[str, Any] = field(default_factory=dict)
    data_train_parquet_fp: str | None = None
    data_validation_parquet_fp: str | None = None
    data_test_parquet_fp: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility."""
        return self.features.get(key, getattr(self, key, None))

    def __setitem__(self, key: str, value: Any):
        """Dict-like setting for backward compatibility."""
        if hasattr(self, key) and key != "features":
            setattr(self, key, value)
        else:
            self.features[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.features or hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.features:
            return self.features[key]
        return getattr(self, key, default)

    def keys(self):
        return list(self.features.keys())

    def items(self):
        return self.features.items()

    def to_dict(self) -> dict:
        """Convert to plain dict for serialization."""
        result = {}
        for key, value in self.features.items():
            if hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        if self.data_train_parquet_fp:
            result["data_train_parquet_fp"] = self.data_train_parquet_fp
        if self.data_validation_parquet_fp:
            result["data_validation_parquet_fp"] = self.data_validation_parquet_fp
        if self.data_test_parquet_fp:
            result["data_test_parquet_fp"] = self.data_test_parquet_fp
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSetMetadata":
        """Construct from a plain dict (backward compatibility)."""
        special_keys = {
            "data_train_parquet_fp",
            "data_validation_parquet_fp",
            "data_test_parquet_fp",
            "data_train_hdf5_fp",
            "data_validation_hdf5_fp",
            "data_test_hdf5_fp",
        }
        metadata = cls()
        for key, value in d.items():
            if key in special_keys:
                if "hdf5" not in key:  # Skip HDF5 paths
                    setattr(metadata, key, value)
            else:
                metadata.features[key] = value
        return metadata
