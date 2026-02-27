"""Ludwig schema utilities - pydantic 2 based.

This module provides the foundation for Ludwig's declarative config system.
All config classes inherit from BaseMarshmallowConfig (a pydantic BaseModel)
and use field factory functions (String, Integer, Float, etc.) that return
pydantic Field() objects.
"""

import copy
import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ACTIVE, COLUMN, LUDWIG_SCHEMA_VALIDATION_POLICY, NAME, PROC_COLUMN, TYPE
from ludwig.error import ConfigValidationError
from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.schema.metadata import COMMON_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.misc_utils import scrub_creds
from ludwig.utils.registry import Registry
from ludwig.utils.torch_utils import activations, initializer_registry

# ============================================================================
# LudwigSchemaField - base class replacing marshmallow fields.Field
# ============================================================================


class LudwigSchemaField:
    """Plain Python base class for Ludwig schema fields.

    Replaces marshmallow fields.Field as the base for TypeSelection, DictMarshmallowField (NestedConfigField), and all
    custom field classes. The contract (get_default_field, _jsonschema_type_mapping, _deserialize) stays identical.
    """

    def __init__(self, **kwargs):
        # Store all keyword arguments as attributes for backward compat
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_default_field(self) -> FieldInfo:
        """Create a pydantic FieldInfo for this field.

        Override in subclasses.
        """
        return Field(default=None)

    def _jsonschema_type_mapping(self):
        """Return a JSON schema dict for this field.

        Override in subclasses.
        """
        return None

    def _deserialize(self, value, attr, data, **kwargs):
        """Deserialize a raw value.

        Override in subclasses.
        """
        return value


logger = logging.getLogger(__name__)

RECURSION_STOP_ENUM = {"weights_initializer", "bias_initializer", "norm_params"}


def ludwig_dataclass(cls):
    """No-op decorator.

    Config classes now inherit directly from BaseMarshmallowConfig (pydantic BaseModel).
    """
    return cls


# TODO: Change to RAISE and update descriptions once we want to enforce strict schemas.
LUDWIG_SCHEMA_VALIDATION_POLICY_VAR = os.environ.get(LUDWIG_SCHEMA_VALIDATION_POLICY, "exclude").lower()


class _SchemaAdapter:
    """Adapts pydantic model to marshmallow-like Schema interface for backward compatibility.

    This allows existing code that calls cls.Schema().load(data), cls.Schema().dump(data), and cls.Schema().fields to
    continue working.
    """

    def __init__(self, cls):
        self._cls = cls

    def __call__(self):
        """Allow Schema()() pattern (double-call)."""
        return self

    def load(self, data):
        """Validate and create a config instance from a dict."""
        return self._cls.model_validate(data)

    def dump(self, data):
        """Serialize a config instance or dict to a plain dict."""
        if isinstance(data, BaseMarshmallowConfig):
            return data.to_dict()
        if isinstance(data, dict):
            try:
                instance = self._cls.model_validate(data)
                return instance.to_dict()
            except Exception:
                return data
        return data

    @property
    def fields(self):
        """Return field info dict (pydantic model_fields)."""
        return self._cls.model_fields


# Sentinel for TypeSelection and DictMarshmallowField metadata markers
class _TypeSelectionMarker:
    """Marker stored in Field.metadata to indicate this field uses TypeSelection dispatch."""

    def __init__(self, type_selection):
        self.type_selection = type_selection


class _NestedConfigMarker:
    """Marker stored in Field.metadata to indicate this field uses DictMarshmallowField dispatch."""

    def __init__(self, cls, allow_none=True):
        self.cls = cls
        self.allow_none = allow_none


ConfigT = Any  # TypeVar("ConfigT", bound="BaseMarshmallowConfig")


def _convert_dataclass_field_to_pydantic(dc_field) -> FieldInfo:
    """Convert a dataclasses.Field to a pydantic FieldInfo.

    This is the bridge that allows old marshmallow-style field definitions
    (using dataclasses.field(metadata={"marshmallow_field": ...})) to work
    with pydantic BaseModel classes during the migration period.
    """
    import dataclasses as _dc

    metadata_list = []
    marshmallow_field = None

    # Extract marshmallow_field from metadata
    if dc_field.metadata:
        marshmallow_field = dc_field.metadata.get("marshmallow_field")
        if marshmallow_field is not None:
            # Store as a marker so model_validator can use it for dispatch
            if isinstance(marshmallow_field, TypeSelection):
                metadata_list.append(_TypeSelectionMarker(marshmallow_field))
            elif isinstance(marshmallow_field, DictMarshmallowField):
                # Check if the subclass overrides _jsonschema_type_mapping
                has_custom_schema = (
                    type(marshmallow_field)._jsonschema_type_mapping
                    is not DictMarshmallowField._jsonschema_type_mapping
                )
                if has_custom_schema:
                    # Store as MarshmallowFieldMarker to preserve custom JSON schema generation
                    metadata_list.append(_MarshmallowFieldMarker(marshmallow_field))
                else:
                    metadata_list.append(_NestedConfigMarker(marshmallow_field.cls, marshmallow_field.allow_none))
            else:
                # Generic marshmallow field - store for reference
                metadata_list.append(_MarshmallowFieldMarker(marshmallow_field))

    # Extract default and create FieldInfo.
    # Note: pydantic 2's Field() does not accept a `metadata` kwarg — set it on the FieldInfo after creation.
    if dc_field.default is not _dc.MISSING:
        fi = Field(default=dc_field.default)
    elif dc_field.default_factory is not _dc.MISSING:
        fi = Field(default_factory=dc_field.default_factory)
    else:
        # No default - this is a required field
        fi = Field()
    if metadata_list:
        fi.metadata = metadata_list
    return fi


class _MarshmallowFieldMarker:
    """Stores a marshmallow field for backward compat during migration."""

    def __init__(self, marshmallow_field):
        self.marshmallow_field = marshmallow_field


class _LudwigModelMeta(type(BaseModel)):
    """Metaclass that bridges marshmallow-dataclass patterns to pydantic 2.

    Handles two key behaviors:
    1. Converts dataclasses.Field objects to pydantic FieldInfo in __new__
    2. Allows class-level access to field defaults via __getattr__
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        import dataclasses as _dc

        annotations = namespace.get("__annotations__", {})

        # Detect @property definitions and prevent pydantic from treating them as field defaults.
        # Properties that don't shadow inherited fields work fine as-is because pydantic
        # only processes annotated attributes. Properties that DO shadow inherited fields
        # should be converted to fields with constant defaults instead (done at the schema
        # class level, not here).
        _saved_properties: dict[str, property] = {}
        for attr_name, value in list(namespace.items()):
            if isinstance(value, property) and attr_name in annotations:
                # A property in this class's own annotations would confuse pydantic.
                # Remove it from annotations (it won't become a field).
                _saved_properties[attr_name] = value
                del namespace[attr_name]
                annotations.pop(attr_name, None)

        # Convert dataclass field() objects and marshmallow field descriptors to pydantic Field()
        for attr_name in list(annotations.keys()):
            if attr_name in namespace:
                value = namespace[attr_name]
                if isinstance(value, _dc.Field):
                    namespace[attr_name] = _convert_dataclass_field_to_pydantic(value)
                elif isinstance(value, LudwigSchemaField) and hasattr(value, "get_default_field"):
                    # TypeSelection and DictMarshmallowField instances need conversion
                    namespace[attr_name] = value.get_default_field()

        # Auto-widen annotations to bridge marshmallow→pydantic gap.
        # In marshmallow, annotations were decorative. In pydantic, they're enforced.
        import types
        import typing

        for attr_name, ann in list(annotations.items()):
            # Skip ClassVar annotations
            origin = getattr(ann, "__origin__", None)
            if origin is typing.ClassVar:
                continue

            if attr_name not in namespace:
                continue

            value = namespace[attr_name]

            # For fields with markers (TypeSelection/DictMarshmallowField/MarshmallowField),
            # set annotation to Any since the actual validation happens in the marker
            if isinstance(value, FieldInfo):
                jse = getattr(value, "json_schema_extra", None)
                has_marker = False
                if isinstance(jse, dict) and "metadata" in jse:
                    has_marker = any(
                        isinstance(m, (_TypeSelectionMarker, _NestedConfigMarker, _MarshmallowFieldMarker))
                        for m in jse["metadata"]
                    )
                for meta in getattr(value, "metadata", None) or []:
                    if isinstance(meta, (_TypeSelectionMarker, _NestedConfigMarker, _MarshmallowFieldMarker)):
                        has_marker = True
                        break

                if has_marker:
                    annotations[attr_name] = Any
                    continue

                # Widen to include None if default is None or enum contains None
                from pydantic_core import PydanticUndefined

                should_widen = value.default is None and value.default is not PydanticUndefined
                if not should_widen:
                    # Also widen if the enum (from allow_none=True in StringOptions etc.) contains None
                    jse_enum = (jse or {}).get("enum") if isinstance(jse, dict) else None
                    if isinstance(jse_enum, list) and None in jse_enum:
                        should_widen = True
                if not should_widen:
                    # Also widen if allow_none=True was explicitly set in the field factory
                    if isinstance(jse, dict) and jse.get("allow_none"):
                        should_widen = True

                if should_widen:
                    is_union = origin in (types.UnionType,)
                    try:
                        is_union = is_union or origin is typing.Union
                    except (AttributeError, TypeError):
                        pass

                    has_none = False
                    if is_union:
                        has_none = type(None) in getattr(ann, "__args__", ())

                    if not has_none:
                        try:
                            annotations[attr_name] = ann | None
                        except TypeError:
                            pass

            elif value is None:
                # Plain None default
                try:
                    annotations[attr_name] = ann | None
                except TypeError:
                    pass

        namespace["__annotations__"] = annotations
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent")
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Restore @property descriptors that we removed from namespace.
        if _saved_properties:
            for pname, prop in _saved_properties.items():
                setattr(cls, pname, prop)

        return cls

    def __getattr__(cls, name: str) -> Any:
        """Allow accessing field defaults as class attributes (e.g., cls.type)."""
        for klass in cls.__mro__:
            pf = vars(klass).get("__pydantic_fields__")
            if pf is not None and isinstance(pf, dict) and name in pf:
                field_info = pf[name]
                from pydantic_core import PydanticUndefined

                if field_info.default is not PydanticUndefined:
                    return field_info.default
                break
        raise AttributeError(name)


@DeveloperAPI
class BaseMarshmallowConfig(BaseModel, metaclass=_LudwigModelMeta):
    """Base pydantic model for all Ludwig config classes.

    Maintains backward-compatible API (from_dict, to_dict, Schema, etc.) while using pydantic 2 internally for
    validation and serialization.
    """

    model_config = ConfigDict(
        extra="ignore" if LUDWIG_SCHEMA_VALIDATION_POLICY_VAR == "exclude" else "forbid",
        arbitrary_types_allowed=True,
        validate_default=False,
        revalidate_instances="never",
        populate_by_name=True,
        strict=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _pre_validate(cls, data: Any) -> Any:
        """Pre-validation: log deprecation warnings, resolve TypeSelection/nested fields."""
        if not isinstance(data, dict):
            return data

        # Log deprecation warnings for unknown fields
        valid_fields = set(cls.model_fields.keys())
        for key in list(data.keys()):
            if key not in valid_fields and key != "type":
                warnings.warn(
                    f'"{key}" is not a valid parameter for the "{cls.__name__}" schema, will be flagged '
                    "as an error in a future version",
                    DeprecationWarning,
                )

        # Resolve TypeSelection, DictMarshmallowField, and legacy marshmallow fields
        for fname, finfo in cls.model_fields.items():
            if fname not in data:
                continue
            value = data[fname]

            # Get markers from both metadata and json_schema_extra
            markers = list(finfo.metadata or [])
            jse = finfo.json_schema_extra
            if isinstance(jse, dict) and "metadata" in jse:
                markers.extend(jse["metadata"])

            for meta in markers:
                if isinstance(meta, _TypeSelectionMarker):
                    data[fname] = meta.type_selection.resolve(value)
                    break
                elif isinstance(meta, _NestedConfigMarker):
                    if isinstance(value, BaseMarshmallowConfig):
                        break  # Already a config instance, skip re-validation
                    if isinstance(value, dict):
                        try:
                            data[fname] = meta.cls.model_validate(value)
                        except Exception as e:
                            raise ConfigValidationError(
                                f"Invalid params: {value}, see `{meta.cls}` definition. Error: {e}"
                            )
                    break
                elif isinstance(meta, _MarshmallowFieldMarker):
                    # Legacy marshmallow field - use its _deserialize for validation
                    # Skip if value is already a config instance (avoid double-validation)
                    if isinstance(value, BaseMarshmallowConfig):
                        break
                    mfield = meta.marshmallow_field
                    if hasattr(mfield, "_deserialize") and value is not None:
                        try:
                            data[fname] = mfield._deserialize(value, fname, data)
                        except Exception as e:
                            # Re-raise ConfigValidationError (from __post_init__) and
                            # from _deserialize rather than swallowing them
                            if isinstance(e, ConfigValidationError):
                                raise
                            pass  # Let pydantic handle other validation errors
                    break

        return data

    @model_validator(mode="after")
    def _validate_field_constraints(self):
        """Post-validation: enforce enum constraints stored in json_schema_extra."""
        for fname, finfo in type(self).model_fields.items():
            value = getattr(self, fname, None)
            extra = finfo.json_schema_extra
            if not isinstance(extra, dict):
                continue

            # Validate enum constraints (from StringOptions, IntegerOptions)
            if "enum" in extra and value is not None:
                allowed = extra["enum"]
                if value not in allowed:
                    raise ValueError(f"Field '{fname}': value {value!r} not in allowed options {allowed}")

            # Validate float tuple range constraints
            if "_float_tuple_range" in extra and value is not None:
                spec = extra["_float_tuple_range"]
                if not isinstance(value, (tuple, list)) or len(value) != spec["n"]:
                    raise ValueError(f"Field '{fname}': expected {spec['n']}-tuple, got {value!r}")
                for v in value:
                    if spec.get("min") is not None and v < spec["min"]:
                        raise ValueError(f"Field '{fname}': value {v} below minimum {spec['min']}")
                    if spec.get("max") is not None and v > spec["max"]:
                        raise ValueError(f"Field '{fname}': value {v} above maximum {spec['max']}")

            # Validate embed field (int or str from options)
            if "_embed_options" in extra and value is not None:
                embed_options = extra["_embed_options"]
                if isinstance(value, str) and value not in embed_options:
                    raise ValueError(f"Field '{fname}': string value {value!r} not in {embed_options}")
                if not isinstance(value, (str, int)):
                    raise ValueError(f"Field '{fname}': expected str, int, or None, got {type(value).__name__}")

            # Validate initializer_or_dict field
            if "_initializer_options" in extra and value is not None:
                init_options = extra["_initializer_options"]
                if isinstance(value, str) and value not in init_options:
                    raise ValueError(f"Field '{fname}': initializer {value!r} not in {init_options}")
                if isinstance(value, dict):
                    if "type" not in value:
                        raise ValueError(f"Field '{fname}': dict must contain 'type' key")
                    if value["type"] not in init_options:
                        raise ValueError(f"Field '{fname}': initializer type {value['type']!r} not in {init_options}")
                if not isinstance(value, (str, dict)):
                    raise ValueError(f"Field '{fname}': expected str or dict, got {type(value).__name__}")

        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting arbitrary attributes on config instances.

        Ludwig code dynamically sets attributes like saved_weights_in_checkpoint, proc_column, etc. on config objects.
        Pydantic 2 normally rejects setting attributes not defined as fields, so we override to allow it.
        """
        try:
            super().__setattr__(name, value)
        except ValueError:
            # Attribute not in model fields - allow it anyway (dataclass behavior)
            object.__setattr__(self, name, value)

    def model_post_init(self, __context: Any) -> None:
        """Bridge: call __post_init__ if defined by subclass (dataclass convention)."""
        super().model_post_init(__context)
        # Check if THIS class (or a parent) defines __post_init__
        post_init = getattr(type(self), "__post_init__", None)
        if post_init is not None:
            post_init(self)

    def to_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of this config.

        Recursively converts nested config objects and scrubs credentials.
        """
        return scrub_creds(convert_submodules(vars(self)))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseMarshmallowConfig":
        """Create a config instance from a dictionary."""
        return cls.model_validate(d)

    @classmethod
    @lru_cache(maxsize=None)
    def get_valid_field_names(cls) -> set[str]:
        """Return the set of valid field names for this config class."""
        return set(cls.model_fields.keys())

    @classmethod
    @lru_cache(maxsize=None)
    def get_class_schema(cls):
        """Return a schema adapter for backward compatibility.

        Returns an object with .load() and .fields methods.
        """
        return _SchemaAdapter(cls)

    @classmethod
    def Schema(cls):
        """Backward compatibility: return a schema adapter with .load(), .dump(), .fields."""
        return _SchemaAdapter(cls)

    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)


@DeveloperAPI
def get_marshmallow_field_class_name(field_info):
    """Returns a human-readable string of the field class name.

    For backward compat, checks both pydantic metadata and marshmallow_field.
    """
    # Check for marshmallow_field in metadata (legacy)
    if hasattr(field_info, "metadata"):
        for meta in field_info.metadata or []:
            if hasattr(meta, "__class__"):
                return meta.__class__.__name__
    # For pydantic FieldInfo, return the annotation name
    if hasattr(field_info, "annotation"):
        return str(field_info.annotation)
    return "Unknown"


@DeveloperAPI
def load_config(cls: type["BaseMarshmallowConfig"], **kwargs) -> "BaseMarshmallowConfig":
    """Takes a config class and instantiates it with the given keyword args as parameters."""
    assert_is_a_marshmallow_class(cls)
    return cls.model_validate(kwargs)


@DeveloperAPI
def load_trainer_with_kwargs(model_type: str, kwargs: dict) -> tuple["BaseMarshmallowConfig", dict[str, Any]]:
    """Special case of `load_config_with_kwargs` for the trainer schemas."""
    from ludwig.constants import MODEL_LLM
    from ludwig.schema.trainer import ECDTrainerConfig, LLMTrainerConfig

    if model_type == MODEL_LLM:
        trainer_schema = LLMTrainerConfig
    else:
        trainer_schema = ECDTrainerConfig

    return load_config_with_kwargs(trainer_schema, kwargs)


@DeveloperAPI
def load_config_with_kwargs(
    cls: type["BaseMarshmallowConfig"], kwargs_overrides
) -> tuple["BaseMarshmallowConfig", dict[str, Any]]:
    """Instantiates a config class filtering kwargs to only valid fields.

    Returns a tuple of (config, remaining_kwargs).
    """
    assert_is_a_marshmallow_class(cls)
    fields = cls.model_fields.keys()
    return load_config(cls, **{k: v for k, v in kwargs_overrides.items() if k in fields}), {
        k: v for k, v in kwargs_overrides.items() if k not in fields
    }


@DeveloperAPI
def convert_submodules(config_dict: dict) -> dict[str, Any]:
    """Helper for converting submodules to dictionaries during config serialization."""
    output_dict = copy.deepcopy(config_dict)

    for k, v in output_dict.items():
        if isinstance(v, dict):
            convert_submodules(v)
        elif isinstance(v, BaseMarshmallowConfig):
            output_dict[k] = v.to_dict()
            convert_submodules(output_dict[k])
        elif isinstance(v, list):
            output_dict[k] = [x.to_dict() if isinstance(x, BaseMarshmallowConfig) else x for x in v]
        elif isinstance(v, ListSerializable):
            output_dict[k] = v.to_list()

    return output_dict


@DeveloperAPI
def create_cond(if_pred: dict, then_pred: dict):
    """Returns a JSONSchema conditional for the given if-then predicates."""
    return {
        "if": {"properties": {k: {"const": v} for k, v in if_pred.items()}},
        "then": {"properties": then_pred},
    }


@DeveloperAPI
def remove_duplicate_fields(properties: dict, fields: list[str] | None = None) -> None:
    """Util function for removing duplicated schema elements."""
    duplicate_fields = [NAME, TYPE, COLUMN, PROC_COLUMN, ACTIVE] if fields is None else fields
    for key in duplicate_fields:
        if key in properties:
            del properties[key]


@DeveloperAPI
class ListSerializable(ABC):
    @abstractmethod
    def to_list(self) -> list:
        pass


@DeveloperAPI
def assert_is_a_marshmallow_class(cls):
    """Assert that cls is a Ludwig config class (pydantic BaseModel)."""
    assert issubclass(
        cls, BaseMarshmallowConfig
    ), f"Expected a Ludwig config class (BaseMarshmallowConfig subclass), but `{cls}` is not."


def _default_matches_json_type(default_val, type_str) -> bool:
    """Check if a default value is consistent with a JSON schema type string.

    Returns True if the default value matches the type string, False otherwise. This is used to avoid emitting 'type':
    'integer' when the default is 7.5 (float), which was a common pattern in the marshmallow era where type enforcement
    was looser.
    """
    if isinstance(type_str, list):
        # Union type like ["integer", "null"]
        return any(_default_matches_json_type(default_val, t) for t in type_str)
    _CHECKS = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, (list, tuple)),
        "null": lambda v: v is None,
    }
    check = _CHECKS.get(type_str)
    if check is None:
        return True  # Unknown type, don't block
    return check(default_val)


def _field_info_to_jsonschema(fname: str, finfo: FieldInfo, annotation: type | None = None) -> dict:
    """Convert a pydantic FieldInfo to a JSON schema fragment.

    Checks metadata markers for TypeSelection/DictMarshmallowField/legacy marshmallow fields, and falls back to type-
    based mapping for plain fields.
    """
    # Check for markers in both metadata and json_schema_extra
    markers = list(finfo.metadata or [])
    jse = finfo.json_schema_extra
    if isinstance(jse, dict) and "metadata" in jse:
        markers.extend(jse["metadata"])

    for meta in markers:
        if isinstance(meta, _TypeSelectionMarker):
            ts = meta.type_selection
            custom = ts._jsonschema_type_mapping()
            if custom is not None:
                return custom
            return {"type": "object"}

        if isinstance(meta, _NestedConfigMarker):
            return unload_jsonschema_from_marshmallow_class(meta.cls)

        if isinstance(meta, _MarshmallowFieldMarker):
            mf = meta.marshmallow_field
            if hasattr(mf, "_jsonschema_type_mapping"):
                custom = mf._jsonschema_type_mapping()
                if custom is not None:
                    return custom
            # Handle FeatureList-style fields with inner and length constraints
            if hasattr(mf, "inner") and mf.inner is not None:
                inner_schema = {}
                if hasattr(mf.inner, "_jsonschema_type_mapping"):
                    inner_schema = mf.inner._jsonschema_type_mapping() or {}
                result = {"type": "array", "items": inner_schema}
                if hasattr(mf, "min_length") and mf.min_length is not None:
                    result["minItems"] = mf.min_length
                if hasattr(mf, "max_length") and mf.max_length is not None:
                    result["maxItems"] = mf.max_length
                return result
            return {"type": "object"}

    # Handle InitializerOrDict fields
    from pydantic_core import PydanticUndefined

    extra = finfo.json_schema_extra
    if isinstance(extra, dict) and "_initializer_options" in extra:
        init_options = extra["_initializer_options"]
        return {
            "oneOf": [
                {"type": "string", "enum": init_options},
                {
                    "type": "object",
                    "properties": {"type": {"type": "string", "enum": init_options}},
                    "required": ["type"],
                    "additionalProperties": True,
                },
                {"type": "null"},
            ],
            "default": finfo.default if finfo.default is not PydanticUndefined else "xavier_uniform",
            "description": finfo.description or "",
        }

    # Build schema from field info
    schema: dict[str, Any] = {}

    # Description
    desc = finfo.description or ""
    if desc:
        schema["description"] = desc

    # Default value
    from pydantic_core import PydanticUndefined

    if finfo.default is not PydanticUndefined:
        if not callable(finfo.default) and not isinstance(finfo.default, property):
            schema["default"] = finfo.default

    # Enum constraint from json_schema_extra
    extra = finfo.json_schema_extra
    if isinstance(extra, dict):
        if "enum" in extra:
            schema["enum"] = extra["enum"]
        if "parameter_metadata" in extra:
            schema["parameter_metadata"] = copy.deepcopy(extra["parameter_metadata"])

    # Always include parameter_metadata (default if not explicitly provided)
    if "parameter_metadata" not in schema:
        schema["parameter_metadata"] = convert_metadata_to_json(None)

    # Map type annotation to JSON schema type
    # Only emit type if annotation and default are consistent (avoid mismatches
    # like annotation=int but default=7.5 which was common in marshmallow era)
    if annotation is not None:
        type_str = _annotation_to_json_type(annotation)
        if type_str:
            # If the enum contains None, the JSON schema type must include "null"
            enum_vals = schema.get("enum")
            if enum_vals is not None and None in enum_vals:
                if isinstance(type_str, list):
                    if "null" not in type_str:
                        type_str = type_str + ["null"]
                elif type_str != "null":
                    type_str = [type_str, "null"]

            # Check for mismatch between annotation type and default value
            from pydantic_core import PydanticUndefined

            default_val = finfo.default if finfo.default is not PydanticUndefined else None
            if default_val is not None and not _default_matches_json_type(default_val, type_str):
                pass  # Skip emitting type to avoid JSON schema validation failures
            else:
                schema["type"] = type_str

    # Range constraints and pattern from pydantic Field metadata
    from annotated_types import Ge, Gt, Le, Lt

    for meta in finfo.metadata or []:
        if isinstance(meta, Ge):
            schema["minimum"] = meta.ge
        elif isinstance(meta, Gt):
            schema["exclusiveMinimum"] = meta.gt
        elif isinstance(meta, Le):
            schema["maximum"] = meta.le
        elif isinstance(meta, Lt):
            schema["exclusiveMaximum"] = meta.lt
        elif hasattr(meta, "pattern") and getattr(meta, "pattern", None) is not None:
            schema["pattern"] = meta.pattern

    return schema


def _annotation_to_json_type(annotation) -> str | list | None:
    """Map a Python type annotation to a JSON schema type string."""
    import types

    origin = getattr(annotation, "__origin__", None)

    # Handle Python 3.10+ union types (e.g. float | None) which are instances of
    # types.UnionType directly, without __origin__
    if isinstance(annotation, types.UnionType):
        args = annotation.__args__
        has_none = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base = _annotation_to_json_type(non_none[0])
            if has_none and base:
                return [base, "null"]
            return base
        return None

    # Also handle typing.Union
    try:
        import typing

        if origin is typing.Union:
            args = annotation.__args__
            has_none = type(None) in args
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                base = _annotation_to_json_type(non_none[0])
                if has_none and base:
                    return [base, "null"]
                return base
            return None
    except (AttributeError, TypeError):
        pass

    _TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        tuple: "array",
    }

    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]

    return None


@DeveloperAPI
def unload_jsonschema_from_marshmallow_class(mclass, additional_properties: bool = True, title: str = None) -> dict:
    """Get a JSON schema dict for a Ludwig config class.

    Iterates over pydantic model_fields and checks metadata markers for TypeSelection, DictMarshmallowField, and legacy
    marshmallow fields.
    """
    assert_is_a_marshmallow_class(mclass)

    properties = {}
    annotations = {}

    # Gather annotations from the class and its MRO
    for klass in reversed(mclass.__mro__):
        annotations.update(getattr(klass, "__annotations__", {}))

    for fname, finfo in mclass.model_fields.items():
        ann = annotations.get(fname)
        properties[fname] = _field_info_to_jsonschema(fname, finfo, ann)

    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if title is not None:
        schema["title"] = title
    return schema


# ============================================================================
# Field Factory Functions
# ============================================================================
# All return pydantic Field() objects (FieldInfo) that can be used as class
# variable defaults in BaseMarshmallowConfig subclasses.
# ============================================================================


def _make_json_schema_extra(
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
    **extra,
) -> dict | None:
    """Build json_schema_extra dict for Field(), returning None if empty."""
    result = {}
    if parameter_metadata:
        result["parameter_metadata"] = convert_metadata_to_json(parameter_metadata)
    result.update(extra)
    return result or None


@DeveloperAPI
def InitializerOptions(default: str = "xavier_uniform", description="", parameter_metadata: ParameterMetadata = None):
    """Utility wrapper that returns a `StringOptions` field with keys from `initializer_registry`."""
    return StringOptions(
        list(initializer_registry.keys()),
        default=default,
        allow_none=False,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def ActivationOptions(default: str | None = "relu", description=None, parameter_metadata: ParameterMetadata = None):
    """Utility wrapper that returns a `StringOptions` field with keys from `activations` registry."""
    description = description or "Default activation function applied to the output of the fully connected layers."
    parameter_metadata = parameter_metadata or COMMON_METADATA["activation"]
    return StringOptions(
        list(activations.keys()),
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def ReductionOptions(default: None | str = None, description="", parameter_metadata: ParameterMetadata = None):
    """Utility wrapper that returns a `StringOptions` field with keys from `reduce_mode_registry`."""
    return StringOptions(
        list(reduce_mode_registry.keys()),
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def RegularizerOptions(
    default: None | str,
    allow_none: bool = False,
    description="",
    parameter_metadata: ParameterMetadata = None,
):
    """Utility wrapper that returns a `StringOptions` field with prefilled regularizer options."""
    return StringOptions(
        ["l1", "l2", "l1_l2"],
        default=default,
        allow_none=allow_none,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def String(
    description: str,
    default: None | str,
    allow_none: bool = False,
    pattern: str = None,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for string values."""
    if not allow_none and default is not None and not isinstance(default, str):
        raise ValueError(f"Provided default `{default}` should be a string!")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    kwargs = {}
    if pattern is not None:
        kwargs["pattern"] = pattern

    return Field(
        default=default,
        description=description,
        json_schema_extra=json_extra,
        **kwargs,
    )


@DeveloperAPI
def StringOptions(
    options: list[str],
    default: None | str,
    allow_none: bool = False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field that enforces string inputs must be one of `options`."""
    options = list(options)  # ensure list, not dict_keys or other iterable
    assert len(options) > 0, "Must provide non-empty list of options!"

    if default is not None:
        assert isinstance(default, str), f"Provided default `{default}` should be a string!"

    if allow_none and None not in options:
        options = options + [None]
    if not allow_none and None in options:
        options = [o for o in options if o is not None]

    assert len(options) == len(
        {o for o in options if o is not None} | ({None} if None in options else set())
    ), f"Provided options must be unique! See: {options}"
    assert default in options, f"Provided default `{default}` is not one of allowed options: {options}"

    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        enum=options,
    )
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def ProtectedString(
    pstring: str,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Alias for a `StringOptions` field with only one option."""
    return StringOptions(
        options=[pstring],
        default=pstring,
        allow_none=False,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def IntegerOptions(
    options: list[int],
    default: None | int,
    allow_none: bool = False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field that enforces integer inputs must be one of `options`."""
    if len(options) <= 0:
        raise ValueError("Must provide non-empty list of options!")
    if default is not None and not isinstance(default, int):
        raise ValueError(f"Provided default `{default}` should be an int!")
    if allow_none and None not in options:
        options = list(options) + [None]
    if not allow_none and None in options:
        options = [o for o in options if o is not None]
    if default not in options:
        raise ValueError(f"Provided default `{default}` is not one of allowed options: {options}")

    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        enum=options,
    )
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def Boolean(default: bool, description: str = "", parameter_metadata: ParameterMetadata = None):
    """Returns a pydantic Field for boolean values."""
    if default is not None and not isinstance(default, bool):
        raise ValueError(f"Invalid default: `{default}`")

    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata)
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def Integer(
    default: None | int,
    allow_none=False,
    description="",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field strictly enforcing integer inputs."""
    if default is not None and not isinstance(default, int):
        raise ValueError(f"Invalid default: `{default}`")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def PositiveInteger(
    description: str,
    default: None | int,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field enforcing positive integer inputs (>= 1)."""
    if default is not None:
        if not isinstance(default, int) or default < 1:
            raise ValueError(f"Invalid default: `{default}`")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, ge=1, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def NonNegativeInteger(
    description: str,
    default: None | int,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field enforcing nonnegative integer inputs (>= 0)."""
    if default is not None:
        if not isinstance(default, int) or default < 0:
            raise ValueError(f"Invalid default: `{default}`")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, ge=0, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def IntegerRange(
    description: str,
    default: None | int,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
    min: int = None,
    max: int = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    """Returns a pydantic Field enforcing integer inputs within a range."""
    if default is not None:
        if not isinstance(default, int):
            raise ValueError(f"Invalid default: `{default}`")
        if min is not None and ((min_inclusive and default < min) or (not min_inclusive and default <= min)):
            raise ValueError(f"Invalid default: `{default}` (below min {min})")
        if max is not None and ((max_inclusive and default > max) or (not max_inclusive and default >= max)):
            raise ValueError(f"Invalid default: `{default}` (above max {max})")

    kwargs = {}
    if min is not None:
        kwargs["ge" if min_inclusive else "gt"] = min
    if max is not None:
        kwargs["le" if max_inclusive else "lt"] = max

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, description=description, json_schema_extra=json_extra, **kwargs)


@DeveloperAPI
def Float(
    default: None | float | int,
    allow_none=False,
    description="",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for float inputs."""
    if default is not None and not isinstance(default, (float, int)):
        raise ValueError(f"Invalid default: `{default}`")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def NonNegativeFloat(
    default: None | float,
    allow_none: bool = False,
    description: str = "",
    max: float | None = None,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field enforcing nonnegative float inputs."""
    if default is not None:
        if not isinstance(default, (float, int)) or default < 0:
            raise ValueError(f"Invalid default: `{default}`")
        if max is not None and default > max:
            raise ValueError(f"Invalid default: `{default}` (above max {max})")

    kwargs = {"ge": 0.0}
    if max is not None:
        kwargs["le"] = max

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, description=description, json_schema_extra=json_extra, **kwargs)


@DeveloperAPI
def FloatRange(
    default: None | float,
    allow_none: bool = False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
    min: int = None,
    max: int = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    """Returns a pydantic Field enforcing float inputs within a range."""
    if default is not None:
        if not isinstance(default, (float, int)):
            raise ValueError(f"Invalid default: `{default}`")

    kwargs = {}
    if min is not None:
        kwargs["ge" if min_inclusive else "gt"] = min
    if max is not None:
        kwargs["le" if max_inclusive else "lt"] = max

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata, **extra_kwargs)
    return Field(default=default, description=description, json_schema_extra=json_extra, **kwargs)


@DeveloperAPI
def Dict(
    default: None | dict = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for dict values."""
    allow_none = allow_none or default is None

    if default is not None:
        if not isinstance(default, dict):
            raise ValueError(f"Invalid default: `{default}`")
        if not all(isinstance(k, str) for k in default.keys()):
            raise ValueError(f"Invalid default: `{default}` (non-string keys)")
    elif not allow_none:
        default = {}

    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata)

    if default is None:
        return Field(default=None, description=description, json_schema_extra=json_extra)
    return Field(default_factory=lambda: copy.deepcopy(default), description=description, json_schema_extra=json_extra)


@DeveloperAPI
def List(
    list_type: type[str] | type[int] | type[float] | type[list] = str,
    inner_type: type[str] | type[int] | type[float] | type[dict] = float,
    default: None | list[Any] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for list values."""
    if default is not None:
        if not isinstance(default, list):
            raise ValueError(f"Invalid default: `{default}`")
    elif not allow_none:
        default = []

    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata)

    if default is None:
        return Field(default=None, description=description, json_schema_extra=json_extra)
    return Field(default_factory=lambda: copy.deepcopy(default), description=description, json_schema_extra=json_extra)


@DeveloperAPI
def DictList(
    default: None | list[dict] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for list-of-dicts values."""
    if default is not None:
        if not isinstance(default, list) or not all(isinstance(d, dict) for d in default):
            raise ValueError(f"Invalid default: `{default}`")
    elif not allow_none:
        default = []

    json_extra = _make_json_schema_extra(description=description, parameter_metadata=parameter_metadata)

    if default is None:
        return Field(default=None, description=description, json_schema_extra=json_extra)
    return Field(default_factory=lambda: copy.deepcopy(default), description=description, json_schema_extra=json_extra)


@DeveloperAPI
def Embed(description: str = "", parameter_metadata: ParameterMetadata = None):
    """Returns a pydantic Field for embedding input feature names (int, str, or None)."""
    _embed_options = ["add"]
    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        _embed_options=_embed_options,
    )
    return Field(default=None, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def InitializerOrDict(
    default: str = "xavier_uniform", description: str = "", parameter_metadata: ParameterMetadata = None
):
    """Returns a pydantic Field allowing str or dict initializer values."""
    initializers = list(initializer_registry.keys())
    if not isinstance(default, str) or default not in initializers:
        raise ValueError(f"Invalid default: `{default}`")

    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        _initializer_options=initializers,
    )
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def FloatRangeTupleDataclassField(
    n: int = 2,
    default: tuple | None = (0.9, 0.999),
    allow_none: bool = False,
    min: int | None = 0,
    max: int | None = 1,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field for an N-dim tuple with values in a range."""
    if default is not None:
        if n != len(default):
            raise ValueError(f"Dimension of tuple '{n}' must match dimension of default val. '{default}'")
        for v in default:
            if min is not None and v < min:
                raise ValueError(f"Invalid default: value {v} below minimum {min}")
            if max is not None and v > max:
                raise ValueError(f"Invalid default: value {v} above maximum {max}")
    if default is None and not allow_none:
        raise ValueError("Default value must not be None if allow_none is False")

    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        _float_tuple_range={"n": n, "min": min, "max": max},
        **extra_kwargs,
    )
    return Field(default=default, description=description, json_schema_extra=json_extra)


@DeveloperAPI
def OneOfOptionsField(
    default: Any,
    description: str,
    field_options: list,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a pydantic Field that accepts values matching any of the field_options.

    Pydantic union validation handles the multi-type dispatch. The field_options are stored in json_schema_extra for
    JSON schema generation.
    """
    extra_kwargs = {}
    if allow_none:
        extra_kwargs["allow_none"] = True
    json_extra = _make_json_schema_extra(
        description=description,
        parameter_metadata=parameter_metadata,
        _oneof_options=True,
        **extra_kwargs,
    )

    if default is None or isinstance(default, (int, str, bool)):
        return Field(default=default, description=description, json_schema_extra=json_extra)
    return Field(default_factory=lambda: copy.deepcopy(default), description=description, json_schema_extra=json_extra)


# ============================================================================
# TypeSelection - Polymorphic config dispatch based on registry
# ============================================================================


class TypeSelection(LudwigSchemaField):
    """Resolves polymorphic config types from a registry based on a key field.

    Used for fields like encoder, decoder, optimizer where the config class depends on a "type" key in the dict value.
    """

    def __init__(
        self,
        registry: Registry,
        default_value: str | None = None,
        key: str = "type",
        description: str = "",
        parameter_metadata: ParameterMetadata = None,
        allow_str_value: bool = False,
        allow_none: bool = False,
        **kwargs,
    ):
        self.registry = registry
        self.default_value = default_value
        self.key = key
        self.allow_str_value = allow_str_value
        self.allow_none = allow_none
        self.description = description
        self.parameter_metadata = parameter_metadata

    def _deserialize(self, value, attr, data, **kwargs):
        """Marshmallow deserialization - delegates to resolve()."""
        return self.resolve(value)

    def resolve(self, value):
        """Resolve a raw value (dict, str, None) to a config instance."""
        if value is None:
            if self.allow_none:
                return None
            return None

        # Already a config instance
        if isinstance(value, BaseMarshmallowConfig):
            return value

        if self.allow_str_value and isinstance(value, str):
            value = self.str_value_to_object(value)

        if isinstance(value, dict):
            cls_type = value.get(self.key)
            cls_type = cls_type.lower() if cls_type else self.default_value
            if cls_type and cls_type in self.registry:
                cls = self.get_schema_from_registry(cls_type)
                try:
                    return cls.model_validate(value)
                except (TypeError, PydanticValidationError) as e:
                    raise ConfigValidationError(f"Invalid params: {value}, see `{cls}` definition") from e
            raise ConfigValidationError(f"Invalid type: '{cls_type}', expected one of: {list(self.registry.keys())}")

        maybe_str = ", `str`," if self.allow_str_value else ""
        raise ConfigValidationError(f"Invalid param {value}, expected `None`{maybe_str} or `dict`")

    def str_value_to_object(self, value: str) -> dict:
        """Convert a string shorthand to a dict with the type key."""
        return {self.key: value}

    def get_schema_from_registry(self, key: str) -> type[BaseMarshmallowConfig]:
        """Look up a config class from the registry."""
        return self.registry[key]

    def get_default_field(self) -> FieldInfo:
        """Create a pydantic Field wrapping this TypeSelection.

        The TypeSelection instance is stored in Field.metadata so the base class's model_validator can use it for
        dispatch.
        """
        if self.default_value is not None:
            cls = self.get_schema_from_registry(self.default_value.lower())
            key = self.key
            dv = self.default_value

            def default_factory(cls=cls, key=key, dv=dv):
                return cls.model_validate({key: dv})

        else:

            def default_factory():
                return None

        fi = Field(default_factory=default_factory)
        fi.metadata = [_TypeSelectionMarker(self)]
        return fi

    def _jsonschema_type_mapping(self):
        """Override in subclass for custom JSON schema."""
        return None


@DeveloperAPI
class DictMarshmallowField(LudwigSchemaField):
    """Validates a dict as a specific config class (non-polymorphic).

    Used for fields where a dict should be deserialized into a fixed config class.
    """

    def __init__(
        self,
        cls: type[BaseMarshmallowConfig],
        allow_none: bool = True,
        default_missing: bool = False,
        description: str = "",
        **kwargs,
    ):
        self.cls = cls
        self.allow_none = allow_none
        self.default_missing = default_missing
        self.description = description

    def _deserialize(self, value, attr, data, **kwargs):
        """Deserialize a dict to a config instance via pydantic model_validate."""
        if value is None:
            return value
        if isinstance(value, dict):
            try:
                return self.cls.model_validate(value)
            except (TypeError, PydanticValidationError) as e:
                raise ConfigValidationError(f"Invalid params: {value}, see `{self.cls}` definition") from e
        raise ConfigValidationError("Field should be None or dict")

    def get_default_field(self) -> FieldInfo:
        """Create a pydantic Field wrapping this DictMarshmallowField."""
        if not self.default_missing:
            cls = self.cls

            def default_factory(cls=cls):
                return cls.model_validate({})

        else:

            def default_factory():
                return None

        # Check if subclass overrides _jsonschema_type_mapping - if so, use
        # MarshmallowFieldMarker to preserve custom JSON schema generation
        has_custom_schema = type(self)._jsonschema_type_mapping is not DictMarshmallowField._jsonschema_type_mapping
        if has_custom_schema:
            marker = _MarshmallowFieldMarker(self)
        else:
            marker = _NestedConfigMarker(self.cls, self.allow_none)

        fi = Field(default_factory=default_factory)
        fi.metadata = [marker]
        return fi

    def _jsonschema_type_mapping(self):
        return unload_jsonschema_from_marshmallow_class(self.cls)


# Backward compatibility aliases
ValidationError = ConfigValidationError
NestedConfigField = DictMarshmallowField
LudwigConfig = BaseMarshmallowConfig
unload_jsonschema_from_config_class = unload_jsonschema_from_marshmallow_class
