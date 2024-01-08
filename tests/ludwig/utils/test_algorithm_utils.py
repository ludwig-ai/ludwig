import pytest

from ludwig.utils.algorithms_utils import topological_sort


@pytest.mark.parametrize(
    "unsorted,sorted",
    [
        (
            [(2, []), (5, [11]), (11, [2, 9, 10]), (7, [11, 8]), (9, []), (10, []), (8, [9]), (3, [10, 8])],
            [(2, []), (9, []), (10, []), (8, [9]), (3, [10, 8]), (11, [2, 9, 10]), (7, [11, 8]), (5, [11])],
        ),
        (
            [("macro", ["action", "contact_type"]), ("contact_type", None), ("action", ["contact_type"])],
            [("contact_type", []), ("action", ["contact_type"]), ("macro", ["action", "contact_type"])],
        ),
    ],
)
def test_topological_sort(unsorted: list, sorted: list) -> None:
    assert topological_sort(unsorted) == sorted
