# Tests the following end-to-end:
#
# 1. W&B is imported
# 2. Overridden methods are called (train_init, train_model, etc.) and run without error
#
# This test runs in an isolated environment because W&B make breaking changes to the
# global interpreter state that will otherwise cause subsequent tests to fail.

import argparse
import os
import shutil
import sys
from unittest.mock import Mock

import ludwig.contrib
from ludwig.contribs.wandb import Wandb

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))

from tests.integration_tests.test_experiment import run_experiment
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import image_feature

parser = argparse.ArgumentParser()
parser.add_argument('--csv-filename', required=True)


def run(csv_filename):
    # enable wandb contrib module
    ludwig.contrib.use_contrib('wandb')

    # Check that wandb has been imported successfully as a contrib package
    contrib_instances = ludwig.contrib.contrib_registry["instances"]
    assert len(contrib_instances) == 1

    wandb_instance = contrib_instances[0]
    assert isinstance(wandb_instance, Wandb)

    # Wrap these methods so we can check that they were called
    wandb_instance.train_init = Mock(side_effect=wandb_instance.train_init)
    wandb_instance.train_model = Mock(side_effect=wandb_instance.train_model)

    # disable sync to cloud
    os.environ['WANDB_MODE'] = 'dryrun'

    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    try:
        # Inputs & Outputs
        input_features = [image_feature(folder=image_dest_folder)]
        output_features = [category_feature()]
        rel_path = generate_data(input_features, output_features, csv_filename)

        # Run experiment
        run_experiment(input_features, output_features, dataset=rel_path)
    finally:
        # Delete the temporary data created
        shutil.rmtree(image_dest_folder, ignore_errors=True)

    # Check that these methods were called at least once
    wandb_instance.train_init.assert_called()
    wandb_instance.train_model.assert_called()


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.csv_filename)
