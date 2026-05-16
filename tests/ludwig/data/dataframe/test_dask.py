import io

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import generate_data_as_dataframe


def test_dask_image_bytes_no_unicode_error():
    """Regression test for GitHub #4149.

    Dask's default dataframe.convert-string:True tries to decode all object-dtype
    columns as UTF-8 strings.  JPEG/PNG bytes start with 0xFF/0x89 — invalid UTF-8
    start bytes — so the conversion raises UnicodeDecodeError.

    Ludwig fixes this by setting dataframe.convert-string:False at import time
    (ludwig/__init__.py), before the caller creates any Dask DataFrame.  The old fix
    in RayBackend.initialize() was too late: user DataFrames are created before
    model.train() is called, so the broken _to_string_dtype node was already baked
    into the task graph.
    """
    import dask
    import dask.dataframe as dd
    from PIL import Image

    from ludwig.data.dataframe.dask import reset_index_across_all_partitions

    assert dask.config.get("dataframe.convert-string") is False, (
        "Ludwig must set dataframe.convert-string:False at import time (ludwig/__init__.py)"
    )

    def _jpeg_bytes() -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, "JPEG")
        return buf.getvalue()

    n = 16
    df = dd.from_pandas(
        pd.DataFrame({"image_data": [_jpeg_bytes() for _ in range(n)], "label": np.arange(n, dtype=float)}),
        npartitions=4,
    )

    # reset_index is called inside Ludwig's build_dataset; it must not raise.
    result = reset_index_across_all_partitions(df)
    computed = result.compute()
    assert len(computed) == n
    assert computed["image_data"].iloc[0][:2] == b"\xff\xd8"  # JPEG magic bytes intact


@pytest.mark.distributed
@pytest.mark.distributed_f
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
