import os
import tempfile

import dask.dataframe as dd
import pandas as pd
import pytest

from ludwig.profiling.dataset_profile import (
    get_column_profile_summaries,
    get_column_profile_summaries_from_proto,
    get_dataset_profile_proto,
    get_dataset_profile_view,
    get_ludwig_type_map_from_column_profile_summaries,
)
from tests.integration_tests.utils import category_feature, generate_data, number_feature


@pytest.fixture(scope="module")
def test_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [
            number_feature(),
            number_feature(),
            category_feature(encoder={"vocab_size": 3}),
            category_feature(encoder={"vocab_size": 3}),
        ]
        output_features = [category_feature(decoder={"vocab_size": 3})]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100
        )
        yield input_features, output_features, dataset_csv


def test_get_dataset_profile_view_works():
    df = pd.DataFrame(
        {
            "animal": ["lion", "shark", "cat", "bear", "jellyfish", "kangaroo", "jellyfish", "jellyfish", "fish"],
            "legs": [4, 0, 4, 4.0, None, 2, None, None, "fins"],
            "weight": [14.3, 11.8, 4.3, 30.1, 2.0, 120.0, 2.7, 2.2, 1.2],
        }
    )

    dataset_profile_view = get_dataset_profile_view(df)
    dataset_profile_view_proto = get_dataset_profile_proto(df, dataset_profile_view)
    column_profile_summaries = get_column_profile_summaries_from_proto(dataset_profile_view_proto)

    assert set(column_profile_summaries.keys()) == {
        "animal",
        "legs",
        "weight",
    }


def test_get_dataset_profile_view_works_dask(test_data):
    input_features, output_features, dataset_csv = test_data
    df = dd.read_csv(dataset_csv)

    dataset_profile_view = get_dataset_profile_view(df)
    dataset_profile_view_proto = get_dataset_profile_proto(df, dataset_profile_view)
    column_profile_summaries = get_column_profile_summaries_from_proto(dataset_profile_view_proto)

    assert len(column_profile_summaries.keys()) == 5


def test_get_ludwig_type_map_from_column_profile_summaries():
    df = pd.DataFrame(
        {
            "animal": ["lion", "shark", "cat", "bear", "jellyfish", "kangaroo", "jellyfish", "jellyfish", "fish"],
            "legs": [4, 0, 4, 4.0, None, 2, None, None, "fins"],
            "weight": [14.3, 11.8, 4.3, 30.1, 2.0, 120.0, 2.7, 2.2, 1.2],
            "account_type": ["human", "bot", "human", "bot", "human", "bot", "human", "bot", "human"],
            "finite_numbers_as_numbers": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "finite_numbers_as_strings": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "handful_of_numbers": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "handful_of_strings": ["human", "bot", "unknown", "human", "bot", "unknown", "human", "bot", "unknown"],
            "bools_as_bools": [True, False, True, False, True, False, True, False, True],
            "bools_as_strings": ["0", "1", "0", "1", "0", "1", "0", "1", "0"],
            "floats_as_strings": ["1.5", "3.7", "2.2", "3.1", "1.8", "1.4", "9.9", "8.7", "9.1"],
        }
    )
    column_profile_summaries = get_column_profile_summaries(df)

    ludwig_type_map = get_ludwig_type_map_from_column_profile_summaries(column_profile_summaries)

    assert ludwig_type_map == {
        "account_type": "binary",
        "animal": "text",
        "bools_as_bools": "binary",
        "bools_as_strings": "binary",
        "finite_numbers_as_numbers": "number",
        "finite_numbers_as_strings": "number",
        "handful_of_numbers": "category",
        "handful_of_strings": "category",
        "floats_as_strings": "number",
        "legs": "number",
        "weight": "number",
    }
