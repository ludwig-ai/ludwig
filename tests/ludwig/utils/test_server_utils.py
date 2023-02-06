import numpy as np

from ludwig.utils.server_utils import NumpyJSONResponse


def test_numpy_json_response():
    response = NumpyJSONResponse({"message": "Ludwig server is up"})

    # Test Python builtin data type encoding.
    assert response.render(None) == b"null"
    assert response.render({}) == b"{}"
    assert response.render(1) == b"1"
    assert response.render(1.0) == b"1.0"
    assert response.render("a") == b'"a"'
    assert response.render([0, 1, 2, 3, 4]) == b"[0,1,2,3,4]"
    assert response.render((0, 1, 2, 3, 4)) == b"[0,1,2,3,4]"
    assert response.render({0, 1, 2, 3, 4}) == b"[0,1,2,3,4]"
    assert response.render({"a": "b"}) == b'{"a":"b"}'

    # Test numpy data type encoding
    for dtype in [np.byte, np.ubyte, np.short, np.ushort, np.int, np.uint, np.longlong, np.ulonglong]:
        x = np.arange(5, dtype=dtype)
        assert response.render(x) == b"[0,1,2,3,4]"
        for i in x:
            assert response.render(i) == f"{i}".encode()

    for dtype in [np.half, np.single, np.double, np.longdouble]:
        x = np.arange(5, dtype=dtype)
        assert response.render(x) == b"[0.0,1.0,2.0,3.0,4.0]"
        for i in x:
            assert response.render(i) == f"{i}".encode()
