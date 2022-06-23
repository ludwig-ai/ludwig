# Tests the following end-to-end:
#
# 1. Comet is imported
# 2. Conflicting modules (i.e., TensorFlow) are not imported
# 3. Overridden methods are called (train_init, train_model, etc.) and run without error
#
# This test runs in an isolated environment to ensure TensorFlow imports are not leaked
# from previous tests.

import argparse
import os
import shutil
import sys
from unittest.mock import Mock, patch

# Comet must be imported before the libraries it wraps
import comet_ml  # noqa

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.contribs.comet import CometCallback

# Bad key will ensure Comet is initialized, but nothing is uploaded externally.
os.environ["COMET_API_KEY"] = "key"

# Add tests dir to the import path
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))

from tests.integration_tests.utils import category_feature, generate_data, image_feature  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--csv-filename", required=True)


def run(csv_filename):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), "generated_images")

    # Inputs & Outputs
    input_features = [image_feature(folder=image_dest_folder)]
    output_features = [category_feature()]
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2},
    }

    callback = CometCallback()
    model = LudwigModel(config, callbacks=[callback])
    output_dir = None

    # Wrap these methods so we can check that they were called
    callback.on_train_init = Mock(side_effect=callback.on_train_init)
    callback.on_train_start = Mock(side_effect=callback.on_train_start)

    with patch("comet_ml.Experiment.log_asset_data") as mock_log_asset_data:
        try:
            # Training with csv
            _, _, output_dir = model.train(dataset=data_csv)
            model.predict(dataset=data_csv)
        finally:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    # Verify that the experiment was created successfully
    assert callback.cometml_experiment is not None

    # Check that these methods were called at least once
    callback.on_train_init.assert_called()
    callback.on_train_start.assert_called()

    # Check that we ran `train_model`, which calls into `log_assert_data`, successfully
    mock_log_asset_data.assert_called()


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.csv_filename)
