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
from unittest.mock import patch, Mock

import ludwig.contrib

# Use contrib module before other imports to avoid importing TensorFlow before comet.
# Bad key will ensure Comet is initialized, but nothing is uploaded externally.
os.environ['COMET_API_KEY'] = 'key'
ludwig.contrib.use_contrib('comet')

from ludwig.api import LudwigModel
from ludwig.contribs.comet import Comet

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))

from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import image_feature

parser = argparse.ArgumentParser()
parser.add_argument('--csv-filename', required=True)


def run(csv_filename):
    # Check that comet has been imported successfully as a contrib package
    contrib_instances = ludwig.contrib.contrib_registry["instances"]
    assert len(contrib_instances) == 1

    comet_instance = contrib_instances[0]
    assert isinstance(comet_instance, Comet)

    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Inputs & Outputs
    input_features = [image_feature(folder=image_dest_folder)]
    output_features = [category_feature()]
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(config)
    output_dir = None

    # Wrap these methods so we can check that they were called
    comet_instance.train_init = Mock(side_effect=comet_instance.train_init)
    comet_instance.train_model = Mock(side_effect=comet_instance.train_model)

    with patch('comet_ml.Experiment.log_asset_data') as mock_log_asset_data:
        try:
            # Training with csv
            _, _, output_dir = model.train(dataset=data_csv)
            model.predict(dataset=data_csv)
        finally:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    # Verify that the experiment was created successfully
    assert comet_instance.cometml_experiment is not None

    # Check that these methods were called at least once
    comet_instance.train_init.assert_called()
    comet_instance.train_model.assert_called()

    # Check that we ran `train_model`, which calls into `log_assert_data`, successfully
    mock_log_asset_data.assert_called()


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.csv_filename)
