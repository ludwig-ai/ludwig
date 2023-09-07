import os

import pytest
import yaml

from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.constants import BATCH_SIZE, TRAINER
from ludwig.globals import DESCRIPTION_FILE_NAME
from ludwig.utils import fs_utils
from ludwig.utils.data_utils import use_credentials
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    minio_test_creds,
    private_param,
    remote_tmpdir,
    sequence_feature,
)

pytestmark = pytest.mark.integration_tests_b


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize(
    "fs_protocol,bucket,creds",
    [("file", None, None), private_param(("s3", "ludwig-tests", minio_test_creds()))],
    ids=["file", "s3"],
)
def test_remote_training_set(csv_filename, fs_protocol, bucket, creds, backend, ray_cluster_2cpu):
    with remote_tmpdir(fs_protocol, bucket) as tmpdir:
        with use_credentials(creds):
            input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
            output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

            train_csv = os.path.join(tmpdir, "training.csv")
            val_csv = os.path.join(tmpdir, "validation.csv")
            test_csv = os.path.join(tmpdir, "test.csv")

            local_csv = generate_data(input_features, output_features, csv_filename)
            fs_utils.upload_file(local_csv, train_csv)
            fs_utils.copy(train_csv, val_csv)
            fs_utils.copy(train_csv, test_csv)

            config = {
                "input_features": input_features,
                "output_features": output_features,
                "combiner": {"type": "concat", "output_size": 14},
                TRAINER: {"epochs": 2, BATCH_SIZE: 128},
            }

            config_path = os.path.join(tmpdir, "config.yaml")
            with fs_utils.open_file(config_path, "w") as f:
                yaml.dump(config, f)

            backend_config = {
                "type": backend,
            }
            backend = initialize_backend(backend_config)

            output_directory = os.path.join(tmpdir, "output")
            model = LudwigModel(config_path, backend=backend)
            _, _, output_run_directory = model.train(
                training_set=train_csv, validation_set=val_csv, test_set=test_csv, output_directory=output_directory
            )

            assert os.path.join(output_directory, "api_experiment_run") == output_run_directory
            assert fs_utils.path_exists(os.path.join(output_run_directory, DESCRIPTION_FILE_NAME))
            assert fs_utils.path_exists(os.path.join(output_run_directory, "training_statistics.json"))
            assert fs_utils.path_exists(os.path.join(output_run_directory, "model"))
            assert fs_utils.path_exists(os.path.join(output_run_directory, "model", "model_weights"))

            model.predict(dataset=test_csv, output_directory=output_directory)

            # Train again, this time the cache will be used
            # Resume from the remote output directory
            model.train(
                training_set=train_csv,
                validation_set=val_csv,
                test_set=test_csv,
                model_resume_path=output_run_directory,
            )
