import os
import shutil

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.benchmarking.resource_usage_tracker import ResourceUsageTracker
from ludwig.constants import TRAINER


def test_resource_usage_tracker(tmpdir):
    train_df = pd.DataFrame(np.random.normal(0, 1, size=(50, 3)), columns=["input_1", "input_2", "output_1"])
    eval_df = pd.DataFrame(np.random.normal(0, 1, size=(10, 3)), columns=["input_1", "input_2", "output_1"])

    config = {
        "input_features": [{"name": "input_1", "type": "number"}, {"name": "input_2", "type": "number"}],
        "output_features": [{"name": "output_1", "type": "number"}],
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 1},
    }

    model = LudwigModel(config=config, backend="local")

    with ResourceUsageTracker(tag="train", output_dir=tmpdir, logging_interval=0.1, num_examples=len(train_df)):
        model.train(
            dataset=train_df,
            output_directory=tmpdir,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )

    with ResourceUsageTracker(tag="eval", output_dir=tmpdir, logging_interval=0.1, num_examples=len(eval_df)):
        model.evaluate(dataset=eval_df)

    assert os.path.exists(os.path.join(tmpdir, "train_resource_usage_metrics.json"))
    assert os.path.exists(os.path.join(tmpdir, "eval_resource_usage_metrics.json"))

    shutil.rmtree(tmpdir)
