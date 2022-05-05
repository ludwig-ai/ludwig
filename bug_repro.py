import os
from pickletools import optimize

import dask.dataframe as dd
import pandas as pd
import numpy as np


from tests.integration_tests.utils import (
    category_feature,
    generate_data,
)


def main():
    input_features = [category_feature(vocab_size=5)]
    output_features = [category_feature(vocab_size=5)]
    os.makedirs("./tmp", exist_ok=True)
    rel_path = generate_data(input_features, output_features, os.path.join("./tmp", "data.csv"))

    df = dd.read_csv(rel_path)

    idx2str = {k: sorted(list(set(df[k]))) for k in df.columns}
    str2idx = {k: {v: i for i, v in enumerate(idx2str[k])} for k in idx2str}

    df = df.persist(optimize_graph=False)

    proc_cols = {f"proc_{k}": df[k].map(str2idx[k]).astype(np.int8) for k in df.columns}
    print(proc_cols)

    dataset = df.index.to_frame(name="tmp").drop(columns=["tmp"])
    for k, v in proc_cols.items():
        print(type(v))
        print(k)
        print("v.dtype")
        print(v.dtype)
        dataset[k] = v
        print("dataset[k].dtype")
        print(dataset[k].dtype)
        # dataset[k] = dataset[k].astype(v.dtype)
        print("-" * 10)

    print(dataset.compute())


if __name__ == "__main__":
    main()
