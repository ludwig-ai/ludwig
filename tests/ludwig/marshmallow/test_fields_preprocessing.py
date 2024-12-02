#! /usr/bin/env python


from ludwig.schema.features.preprocessing.binary import BinaryPreprocessingConfig
from ludwig.schema.features.preprocessing.category import CategoryPreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_preprocessing_dataclass_field():
    binary_preproc_dataclass = PreprocessingDataclassField("binary")
    assert binary_preproc_dataclass.default_factory is not None
    assert get_marshmallow_from_dataclass_field(binary_preproc_dataclass).allow_none is False
    assert binary_preproc_dataclass.default_factory() == BinaryPreprocessingConfig()

    category_preproc_dataclass = PreprocessingDataclassField("category")
    assert category_preproc_dataclass.default_factory is not None
    assert get_marshmallow_from_dataclass_field(category_preproc_dataclass).allow_none is False
    assert category_preproc_dataclass.default_factory() == CategoryPreprocessingConfig()
