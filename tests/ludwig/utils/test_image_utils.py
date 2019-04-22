import numpy as np
import pytest

from ludwig.utils.image_utils import num_channels_in_image, get_abs_path


image_2d = np.random.randint(0, 1, (10, 10))
image_3d = np.random.randint(0, 1, (10, 10, 3))


def test_num_channels_in_image():
    assert num_channels_in_image(image_2d) == 1
    assert num_channels_in_image(image_3d) == 3

    with pytest.raises(ValueError):
        num_channels_in_image(np.arange(5))
        num_channels_in_image(None)


def test_get_abs_path():
    assert get_abs_path('a', 'b.jpg') == 'a/b.jpg'
    assert get_abs_path(None, 'b.jpg') == 'b.jpg'
