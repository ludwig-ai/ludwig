import pytest
import uuid
import os


@pytest.fixture()
def csv_filename():
    """
    This methods returns a random filename for the tests to use for generating
    temporary data. After the data is used, all the temporary data is deleted.
    :return: None
    """
    csv_filename = uuid.uuid4().hex[:10].upper() + '.csv'
    yield csv_filename

    delete_temporary_data(csv_filename)


def delete_temporary_data(csv_path):
    """
    Helper method to delete temporary data created for running tests. Deletes
    the csv and hdf5/json data (if any)
    :param csv_path: path to the csv data file
    :return: None
    """
    if os.path.exists(csv_path):
        os.remove(csv_path)

    json_path = os.path.splitext(csv_path)[0] + '.json'
    if os.path.exists(json_path):
        os.remove(json_path)

    hdf5_path = os.path.splitext(csv_path)[0] + '.hdf5'
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
