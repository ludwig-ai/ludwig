import os
import shutil
import tempfile

from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from tests.integration_tests.utils import binary_feature, category_feature, generate_data, numerical_feature


def test_lightgbm(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [numerical_feature(), category_feature(reduce_output="sum")]
        output_features = [binary_feature()]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "fc_size": 14},
            "training": {"epochs": 2},
        }

        backend_config = {
            "type": "local",
        }
        backend = initialize_backend(backend_config)

        model = LudwigModel(config, backend=backend)
        _, _, output_directory = model.train(
            training_set=data_csv,
            validation_set=val_csv,
            test_set=test_csv,
            output_directory=outdir,
        )
        model.predict(dataset=test_csv, output_directory=output_directory)
