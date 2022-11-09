import pandas as pd
import pytest

from ludwig.profiling.dataset_profile import (
    get_column_profile_summaries_from_proto,
    get_dataset_profile_proto,
    get_dataset_profile_view,
)
from tests.integration_tests.utils import category_feature, generate_data_as_dataframe, number_feature


def test_get_dataset_profile_view_works():
    df = pd.DataFrame(
        {
            "animal": ["lion", "shark", "cat", "bear", "jellyfish", "kangaroo", "jellyfish", "jellyfish", "fish"],
            "legs": [4, 0, 4, 4.0, None, 2, None, None, "fins"],
            "weight": [14.3, 11.8, 4.3, 30.1, 2.0, 120.0, 2.7, 2.2, 1.2],
        }
    )

    dataset_profile_view = get_dataset_profile_view(df)
    dataset_profile_view_proto = get_dataset_profile_proto(dataset_profile_view)
    column_profile_summaries = get_column_profile_summaries_from_proto(dataset_profile_view_proto)

    assert set(column_profile_summaries.keys()) == {
        "animal",
        "legs",
        "weight",
    }


@pytest.mark.distributed
def test_get_dataset_profile_view_works_dask():
    import dask.dataframe as dd

    input_features = [
        number_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3})]
    dataset = generate_data_as_dataframe(input_features, output_features, num_examples=100)
    df = dd.from_pandas(dataset, npartitions=5)

    dataset_profile_view = get_dataset_profile_view(df)
    dataset_profile_view_proto = get_dataset_profile_proto(dataset_profile_view)
    column_profile_summaries = get_column_profile_summaries_from_proto(dataset_profile_view_proto)

    assert len(column_profile_summaries.keys()) == 5
