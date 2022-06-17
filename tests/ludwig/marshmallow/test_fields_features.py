#! /usr/bin/env python
from typing import Optional

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass

import ludwig.schema.features as lsf
import ludwig.schema.features.preprocessing as lsp
from ludwig.schema import utils as schema_utils


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_PreprocessingDataclassField():
    binary_preproc_dataclass = lsp.PreprocessingDataclassField('binary')
    assert binary_preproc_dataclass.default_factory is not None
    assert get_marshmallow_from_dataclass_field(binary_preproc_dataclass).allow_none is False
    assert binary_preproc_dataclass.default_factory() == lsp.BinaryPreprocessingConfig()

    category_preproc_dataclass = lsp.PreprocessingDataclassField('category')
    assert category_preproc_dataclass.default_factory is not None
    assert get_marshmallow_from_dataclass_field(category_preproc_dataclass).allow_none is False
    assert category_preproc_dataclass.default_factory() == lsp.CategoryPreprocessingConfig()