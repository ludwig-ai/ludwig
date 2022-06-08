#!/usr/bin/env python

import logging
import os
import shutil

from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.datasets import adult_census_income

shutil.rmtree("./results", ignore_errors=True)

backend_config = {
    "type": "ray",
    "processor": {
        "parallelism": 6,
        "type": "dask",
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 3,
        "resources_per_worker": {
            "CPU": 2,
            "GPU": 0,
        },
    },
}
backend = initialize_backend(backend_config)
model = LudwigModel(config="./config.yaml", logging_level=logging.INFO, backend=backend)

df = adult_census_income.load(split=False)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=df,
    skip_save_processed_input=True,
)

print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
