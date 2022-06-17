#! /usr/bin/env python
from typing import Optional

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass

import ludwig.schema.features as lsf
from ludwig.schema import utils as schema_utils


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_BinaryInputFeature():
    default_binary_input = lsf.BinaryInputFeatureConfig()
    assert default_binary_input.type is not None

    default_binary_output = lsf.BinaryOutputFeatureConfig()
    assert default_binary_output.type is not None


def test_CategoryInputFeature():
    default_category_input = lsf.CategoryInputFeatureConfig()

    default_category_output = lsf.CategoryOutputFeatureConfig()


def test_NumberInputFeature():
    default_number_input = lsf.NumberInputFeatureConfig()

    default_number_output = lsf.NumberOutputFeatureConfig()


def test_SequenceInputFeature():
    default_sequence_input = lsf.SequenceInputFeatureConfig()

    default_sequence_output = lsf.SequenceOutputFeatureConfig()


def test_SetInputFeature():
    default_set_input = lsf.SetInputFeatureConfig()

    default_set_output = lsf.SetOutputFeatureConfig()


def test_TextInputFeature():
    default_text_input = lsf.TextInputFeatureConfig()

    default_text_output = lsf.TextOutputFeatureConfig()


def test_AudioInputFeature():
    default_audio_field = lsf.AudioInputFeatureConfig()


def test_ImageInputFeature():
    default_image_field = lsf.ImageInputFeatureConfig()


def test_BagInputFeature():
    default_bag_field = lsf.BagInputFeatureConfig()


def test_DateInputFeature():
    default_date_field = lsf.DateInputFeatureConfig()


def test_H3InputFeature():
    default_h3_field = lsf.H3InputFeatureConfig()


def test_TimeseriesInputFeature():
    default_timeseries_field = lsf.TimeseriesInputFeatureConfig()


def test_VectorInputFeature():
    default_vector_field = lsf.VectorInputFeatureConfig()

