#! /usr/bin/env python
#
# Checks all dataset download links (just those with URLs, not including kaggle datasets)."""
#
import logging

import requests

import ludwig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_links():
    # Iterate through all datasets, ensure links are valid and reachable.
    all_datasets = ludwig.datasets.list_datasets()

    for dataset_name in all_datasets:
        config = ludwig.datasets._get_dataset_config(dataset_name)
        download_urls = [config.download_urls] if isinstance(config.download_urls, str) else config.download_urls
        for url in download_urls:
            logger.info(f"Checking {dataset_name}: {url}")
            response = requests.head(url)
            assert response.ok, f"Failed to download {dataset_name} from {url}"
