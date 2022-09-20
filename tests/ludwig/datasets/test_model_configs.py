import ludwig.datasets


def test_default_model_config(tmpdir):
    titanic_configs = ludwig.datasets.model_configs_for_dataset("titanic")
    assert len(titanic_configs) > 0

    titanic = ludwig.datasets.get_dataset("titanic", cache_dir=tmpdir)
    assert titanic.default_model_config is not None

    assert titanic.default_model_config == titanic_configs["default"]


def test_dataset_has_no_model_configs(tmpdir):
    bbc_news_configs = ludwig.datasets.model_configs_for_dataset("bbcnews")
    assert len(bbc_news_configs) == 0

    bbcnews = ludwig.datasets.get_dataset("bbcnews", cache_dir=tmpdir)
    assert bbcnews.default_model_config is None
