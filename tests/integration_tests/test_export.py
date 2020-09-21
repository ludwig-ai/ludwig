import os.path
import logging
import pytest

from ludwig.train import train_cli
from ludwig.export import export_savedmodel, export_neuropod
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature


@pytest.mark.parametrize('export_type',
                         ['savedmodel', 'neuropod'])
def test_export(export_type, csv_filename, tmpdir):
    # create location to create training data
    training_set = str(tmpdir.join(csv_filename))
    output_directory = str(tmpdir.join('results'))

    # create training data
    input_features = [numerical_feature(), binary_feature(), category_feature()]
    output_features = [category_feature()]
    training_set = generate_data(
        input_features,
        output_features,
        filename=training_set
    )

    # create model definition
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {
            'epochs': 2
        }
    }

    # train a model
    train_cli(
        model_definition,
        training_set=training_set,
        output_directory=output_directory,
        logging_level=logging.ERROR
    )

    if export_type == 'savedmodel':
        export_location = str(tmpdir.join('export_directory'))

        export_savedmodel(
            os.path.join(output_directory, 'experiment_run', 'model'),
            output_path=export_location
        )

        # check whether required files/directories are present
        assert os.path.isdir(export_location)
        assert os.path.isfile(
            os.path.join(export_location, 'saved_model.pb')
        )
        assert os.path.isdir(
            os.path.join(export_location, 'assets')
        )
        assert os.path.isdir(
            os.path.join(export_location, 'variables')
        )
    elif export_type == 'neuropod':
        export_location = str(tmpdir.join('exported_neuropod_model'))

        export_neuropod(
            os.path.join(output_directory, 'experiment_run', 'model'),
            output_path=export_location
        )

        # check to see if neuropod export created expected file
        os.path.isfile(export_location)
    else:
        ValueError(
            "Invalid export_type specified.  Valid values: "
            "'savedmodel or 'neuropod'"
        )


    print('here')
