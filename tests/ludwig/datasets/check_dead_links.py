#! /usr/bin/env python
#
# Checks all dataset download links (just those with URLs, not including kaggle datasets)."""
#
import logging

import requests

import ludwig.datasets

logger = logging.getLogger(__name__)


def check_link(url):
    response = requests.get(url)
    if response.status_code == 200:
        logger.info(f"OK: {url}")
        return True
    else:
        logger.info(f"{response.status_code}: {url}")
        logger.error(response)
        return False


def test_dead_links():
    # Iterate through all datasets, ensure links are valid and reachable.
    all_datasets = ludwig.datasets.list_datasets()

    for dataset_name in all_datasets:
        config = ludwig.datasets.get_dataset_config(dataset_name)
        download_urls = [config.download_urls] if isinstance(config.download_urls, str) else config.download_urls
        for url in download_urls:
            try:
                check_link(url)
            except Exception as e:
                logger.exception(f"Exception thrown downloading {dataset_name} dataset", e)


if __name__ == "__main__":
    test_dead_links()
