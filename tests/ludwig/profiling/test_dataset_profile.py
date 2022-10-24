import pandas as pd
import pytest

from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.dataset_profile import (
    get_column_profile_summaries,
    get_column_profile_summaries_from_proto,
    get_dataset_profile_proto,
    get_dataset_profile_view,
    get_distinct_values,
    get_distinct_values_balance,
    get_ludwig_type_map_from_column_profile_summaries,
    get_num_distinct_values,
    get_num_nonnull_values,
    get_pct_distinct_values,
    get_pct_null_values,
)


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


def test_get_column_profile_attributes():
    df = pd.DataFrame(
        {
            "animal": ["lion", "shark", "cat", "bear", "jellyfish", "kangaroo", "jellyfish", "jellyfish", "fish"],
            "legs": [4, 0, 4, 4.0, None, 2, None, None, "fins"],
            "weight": [14.3, 11.8, 4.3, 30.1, 2.0, 120.0, 2.7, 2.2, 1.2],
            "account_type": ["human", "bot", "human", "bot", "human", "bot", "human", "bot", "human"],  # Category
            "finite_numbers_as_numbers": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Category
            "finite_numbers_as_strings": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],  # Category
            "bools_as_bools": [True, False, True, False, True, False, True, False, True],  # Binary
            "bools_as_strings": ["0", "1", "0", "1", "0", "1", "0", "1", "0"],  # Binary
            "floats_as_strings": ["1.5", "3.7", "2.2", "3.1", "1.8", "1.4", "9.9", "8.7", "9.1"],  # Number
        }
    )
    column_profile_summaries = get_column_profile_summaries(df)

    assert get_num_distinct_values(column_profile_summaries["animal"]) == 7
    assert get_distinct_values(column_profile_summaries["animal"]) == {
        "fish",
        "kangaroo",
        "jellyfish",
        "cat",
        "bear",
        "shark",
        "lion",
    }
    # None doesn't count towards distinct values.
    assert get_num_distinct_values(column_profile_summaries["legs"]) == 4
    assert get_distinct_values(column_profile_summaries["legs"]) == {"2.000000", "0.000000", "fins", "4.000000"}

    # True fractionals have no frequent items.
    assert get_distinct_values(column_profile_summaries["weight"]) == {}
    # Fractionals, as strings, do have frequent items.
    assert get_distinct_values(column_profile_summaries["floats_as_strings"]) == {
        "1.5",
        "1.4",
        "9.1",
        "9.9",
        "8.7",
        "3.7",
        "2.2",
        "1.8",
        "3.1",
    }
    # True integers have distinct values.
    assert get_distinct_values(column_profile_summaries["finite_numbers_as_numbers"]) == {
        "2",
        "7",
        "3",
        "1",
        "0",
        "6",
        "4",
        "8",
        "5",
    }
    # Integers as strings have distinct values.
    assert get_distinct_values(column_profile_summaries["finite_numbers_as_strings"]) == {
        "8",
        "7",
        "2",
        "5",
        "0",
        "1",
        "6",
        "3",
        "4",
    }

    # Booleans as booleans have no frequent items.
    assert get_distinct_values(column_profile_summaries["bools_as_bools"]) == {}

    # Booleans as strings have frequent items.
    assert get_distinct_values(column_profile_summaries["bools_as_strings"]) == {"0", "1"}

    assert get_distinct_values_balance(column_profile_summaries["animal"]) == pytest.approx(0.3333333333)
    assert get_num_nonnull_values(column_profile_summaries["animal"]) == 9
    assert get_num_nonnull_values(column_profile_summaries["legs"]) == 6
    assert get_pct_null_values(column_profile_summaries["animal"]) == 0
    assert get_pct_null_values(column_profile_summaries["legs"]) == pytest.approx(0.3333333333)
    assert get_pct_distinct_values(column_profile_summaries["bools_as_strings"]) == pytest.approx(0.222222222)


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


def test_dataset_profile_works():
    dataset_profile = dataset_profile_pb2.DatasetProfile()
    dataset_profile.num_examples = 10

    from_serialized = dataset_profile_pb2.DatasetProfile()
    from_serialized.ParseFromString(dataset_profile.SerializeToString())

    assert from_serialized.num_examples == 10
