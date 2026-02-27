import pandas as pd
import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import generate_data_as_dataframe


@pytest.mark.distributed
def test_from_ray_dataset_empty(tmpdir, ray_cluster_2cpu):
    import dask.dataframe as dd

    # Verifies that when the dataset is an empty MapBatches(BatchInferModel), we mitigate Ray's native to_dask()
    # IndexError.
    config = {
        "input_features": [
            {"name": "cat1", "type": "category", "vocab_size": 2},
            {"name": "num1", "type": "number"},
        ],
        "output_features": [
            {"name": "bin1", "type": "binary"},
        ],
        "trainer": {"epochs": 1},
    }
    train_input_df = generate_data_as_dataframe(config["input_features"], config["output_features"])
    model = LudwigModel(config, backend="ray")
    model.train(
        train_input_df,
        output_directory=tmpdir,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_processed_output=True,
        skip_save_processed_input=True,
    )

    predict_input_df = dd.from_pandas(pd.DataFrame([], columns=["cat1", "num1", "bin1"]), npartitions=1)
    model.predict(predict_input_df)
