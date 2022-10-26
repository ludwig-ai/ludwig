import pandas as pd

from ludwig.data.negative_sampling import negative_sample


def test_negative_sample():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": ["a", "b", "b", "c", "a"],
            "label": [1, 1, 1, 1, 1],
        }
    )

    df_with_samples = negative_sample(df, "user_id", "item_id", "label")

    assert 9 <= len(df_with_samples) <= 10
    assert df_with_samples["label"].sum() == 5

    # Check data types
    assert df_with_samples["user_id"].dtype == "int64"
    assert df_with_samples["item_id"].dtype == "object"

    # Check that the negative samples are unique user-item pairs
    assert len(df_with_samples) == len(df_with_samples.drop_duplicates(["user_id", "item_id"]))
