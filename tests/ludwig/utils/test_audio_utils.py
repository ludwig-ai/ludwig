import pytest

from ludwig.utils.audio_utils import is_audio_score


@pytest.mark.parametrize(
    "path, score",
    [
        ("data.wav", 1),
        ("/home/peter/file.amb", 1),
        ("my.mp3", 1),
        ("data.ogg", 1),
        ("data.vorbis", 1),
        ("data.flac", 1),
        ("data.opus", 1),
        ("data.sphere", 1),
        ("video.mp4", 0),
        ("image.png", 0),
        (".wav/image.png", 0),
    ],
)
def test_is_audio_score(path: str, score: int):
    assert is_audio_score(path) == score
