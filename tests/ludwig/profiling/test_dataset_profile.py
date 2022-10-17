from ludwig.datasets import adult_census_income
from ludwig.profiling.dataset_profile import (
    get_column_profile_views_from_proto,
    get_dataset_profile_proto,
    get_dataset_profile_view,
)


def test_get_dataset_profile_view_works():
    train_df, _, _ = adult_census_income.load(split=True)

    dataset_profile_view = get_dataset_profile_view(train_df)
    dataset_profile_view_proto = get_dataset_profile_proto(dataset_profile_view)
    column_profile_views = get_column_profile_views_from_proto(dataset_profile_view_proto)

    assert set(column_profile_views.keys()) == {
        "hours-per-week",
        "income",
        "race",
        "capital-loss",
        "sex",
        "capital-gain",
        "occupation",
        "relationship",
        "native-country",
        "workclass",
        "education-num",
        "age",
        "education",
        "marital-status",
        "fnlwgt",
    }
