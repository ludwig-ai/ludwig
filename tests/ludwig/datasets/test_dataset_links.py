#! /usr/bin/env python
#
# Checks all dataset download links (just those with URLs, not including kaggle datasets)."""
#
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor

import pytest
import requests

import ludwig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_links():
    # Iterate through all datasets, ensure links are valid and reachable.
    all_datasets = ludwig.datasets.list_datasets()

    tasks = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for dataset_name in all_datasets:
            config = ludwig.datasets._get_dataset_config(dataset_name)
            download_urls = [config.download_urls] if isinstance(config.download_urls, str) else config.download_urls
            for url in download_urls:
                future = executor.submit(_check_url, dataset_name, url)
                tasks[future] = (dataset_name, url)

        failures = []
        for future in as_completed(tasks):
            dataset_name, url = tasks[future]
            error = future.result()
            if error:
                failures.append(error)

    assert not failures, "Failed URLs:\n" + "\n".join(failures)


def _check_url(dataset_name, url):
    logger.info(f"Checking {dataset_name}: {url}")
    try:
        response = requests.head(url, timeout=30)
        if not response.ok:
            return f"Failed to download {dataset_name} from {url} (status {response.status_code})"
    except requests.RequestException as e:
        return f"Failed to download {dataset_name} from {url} ({e})"
    return None
