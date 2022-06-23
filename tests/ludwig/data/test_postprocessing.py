import torch

from ludwig.data.postprocessing import convert_dict_to_df


def test_convert_dict_to_df():
    d = {
        "binary_C82EB": {
            "predictions": torch.tensor([True, True, True, False]),
            "probabilities": torch.tensor([[0.4777, 0.5223], [0.4482, 0.5518], [0.4380, 0.5620], [0.5059, 0.4941]]),
        },
        "category_1491D": {
            "predictions": ["NkNUG", "NkNUG", "NkNUG", "NkNUG"],
            "probabilities": torch.tensor(
                [
                    [0.1058, 0.4366, 0.1939, 0.2637],
                    [0.0816, 0.4807, 0.1978, 0.2399],
                    [0.0907, 0.4957, 0.1829, 0.2308],
                    [0.0728, 0.5015, 0.1900, 0.2357],
                ]
            ),
        },
        "num_7B25F": {"predictions": torch.tensor([2.0436, 2.1158, 2.1222, 2.1964])},
    }

    df = convert_dict_to_df(d)

    assert df.shape == (4, 5)
    # Check that all elements in nested lists are stored in each row
    assert all(len(row) == 2 for row in df["binary_C82EB_probabilities"])
    assert all(len(row) == 4 for row in df["category_1491D_probabilities"])
