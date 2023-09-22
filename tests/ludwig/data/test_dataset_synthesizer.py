from ludwig.data import dataset_synthesizer


def test_build_synthetic_dataset(tmpdir):
    features = [
        {"name": "text", "type": "text"},
        {"name": "category", "type": "category"},
        {"name": "number", "type": "number"},
        {"name": "binary", "type": "binary"},
        {"name": "set", "type": "set"},
        {"name": "bag", "type": "bag"},
        {"name": "sequence", "type": "sequence"},
        {"name": "timeseries", "type": "timeseries"},
        {"name": "date", "type": "date"},
        {"name": "h3", "type": "h3"},
        {"name": "vector", "type": "vector"},
        {"name": "audio", "type": "audio"},
        {"name": "image", "type": "image"},
    ]
    assert len(list(dataset_synthesizer.build_synthetic_dataset(100, features, tmpdir))) == 101  # Extra for the header.
