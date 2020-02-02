import logging
import os
import shutil
import sys

import ludwig.contrib
from tests.integration_tests.test_experiment import run_experiment
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data

import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


def test_wandb_experiment(csv_filename):
    # Test W&B integration

    # add wandb arg and detect flag
    sys.argv.append('--wandb')
    ludwig.contrib.contrib_import()

    # disable sync to cloud
    os.environ['WANDB_MODE'] = 'dryrun'

    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Inputs & Outputs
    input_features = [image_feature(folder=image_dest_folder)]
    output_features = [category_feature()]
    rel_path = generate_data(input_features, output_features, csv_filename)

    # Run experiment
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Check a W&B run was created
    assert wandb.run is not None

    # End session
    wandb.join()

    # Remove instance from contrib_registry
    ludwig.contrib.contrib_registry['instances'].pop()

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


if __name__ == '__main__':
    """
    To run tests individually, run:
    ```python -m pytest tests/integration_tests/test_contrib_wandb.py::test_name```
    """
    pass
