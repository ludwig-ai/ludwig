import pickle

from ludwig.error import ConfigValidationError, InputDataError


def test_input_data_error_serializeable():
    err = InputDataError(
        "location", "category", "At least 2 distinct values are required, column only contains ['here']"
    )

    loaded_err: InputDataError = pickle.loads(pickle.dumps(err))

    assert loaded_err.column_name == err.column_name
    assert loaded_err.feature_type == err.feature_type
    assert loaded_err.message == err.message
    assert str(err) == str(loaded_err)


def test_config_validation_error_serializeable():
    err = ConfigValidationError(message="At least 2 distinct values are required, column only contains ['here']")

    loaded_err: ConfigValidationError = pickle.loads(pickle.dumps(err))

    assert loaded_err.message == err.message
    assert str(err) == str(loaded_err)
