import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


def test_get_config_and_load(tmpdir):
    yosemite_config = ludwig.datasets._get_dataset_config("yosemite")
    assert isinstance(yosemite_config, DatasetConfig)

    yosemite_dataset = ludwig.datasets.get_dataset("yosemite", cache_dir=tmpdir)
    assert isinstance(yosemite_dataset, DatasetLoader)
    df = yosemite_dataset.load()
    assert df is not None
    assert len(df) == 18721  # Expected number of rows in Yosemite temperatures dataset.


def test_get_config_kaggle(tmpdir):
    twitter_bots_config = ludwig.datasets._get_dataset_config("twitter_bots")
    assert isinstance(twitter_bots_config, DatasetConfig)

    twitter_bots_dataset = ludwig.datasets.get_dataset("twitter_bots", cache_dir=tmpdir)
    # Twitter bots dataset is large, so we won't load it in this unit test.
    assert isinstance(twitter_bots_dataset, DatasetLoader)
    assert twitter_bots_dataset.is_kaggle_dataset
