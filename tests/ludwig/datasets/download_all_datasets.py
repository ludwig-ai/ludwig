#! /usr/bin/env python
#
# Lists and downloads all datasets, including Kaggle datasets, into ./download_datasets

# You must have valid kaggle credentials in your environment, a few GB of disk space, and good internet bandwidth.
# Also, for each dataset associated with a Kaggle competition you'll need to sign in to Kaggle and accept the terms of
# the competition.
#
from ludwig import datasets


def download_all_datasets():
    """Downloads all datasets to ./downloaded_datasets."""
    dataset_names = datasets.list_datasets()

    print("Datasets: ")
    for name in dataset_names:
        print(f"  {name}")
    print("Downloading all datasets")

    # Download All Datasets
    for dataset_name in dataset_names:
        print(f"Downloading {dataset_name}")
        datasets.download_dataset(dataset_name, "./downloaded_datasets")


if __name__ == "__main__":
    download_all_datasets()
