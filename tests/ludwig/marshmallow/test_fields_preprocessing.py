#! /usr/bin/env python


import ludwig.schema.preprocessing as lsp


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