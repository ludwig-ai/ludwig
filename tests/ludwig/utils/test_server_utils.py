import numpy as np
import pytest

from ludwig.utils.server_utils import NumpyJSONResponse


def test_numpy_json_response():
    response = NumpyJSONResponse()

    # Test Python builtin data type encoding.
    assert response.render(None) == "null".encode("utf-8")
    assert response.render({}) == "{}".encode("utf-8")
    assert response.render(1) == "1".encode("utf-8")
    assert response.render(1.0) == "1.0".encode("utf-8")
    assert response.render("a") == '"a"'.encode("utf-8")
    assert response.render([0, 1, 2, 3, 4]) == "[0,1,2,3,4]".encode("utf-8")
    assert response.render((0, 1, 2, 3, 4)) == "[0,1,2,3,4]".encode("utf-8")
    assert response.render({0, 1, 2, 3, 4}) == "[0,1,2,3,4]".encode("utf-8")
    assert response.render({"a": "b"}) == '{"a":"b"}'.encode("utf-8")

    # Test numpy data type encoding
    for dtype in [np.byte, np.ubyte, np.short, np.ushort, np.int, np.uint, np.longlong, np.ulonglong]:
        x = np.arange(5, dtype=dtype)
        assert response.render(x) == "[0,1,2,3,4]".encode("utf-8")
        for i in x:
            assert response.render(i) == f"{i}".encode("utf-8")

    for dtype in [np.half, np.single, np.double, np.longdouble]:
        x = np.arange(5, dtype=dtype)
        assert response.render(x) == "[0.0,1.0,2.0,3.0,4.0]".encode("utf-8")
        for i in x:
            assert response.render(i) == f"{i}".encode("utf-8")
