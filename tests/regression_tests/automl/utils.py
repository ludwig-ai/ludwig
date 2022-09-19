from pathlib import Path

import ludwig.datasets
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

# Subset of Ludwig Dataset Zoo used for AutoML type inference regression tests.
TEST_DATASET_REGISTRY = {"adult_census_income", "mnist"}


def get_dataset_golden_types_path(dataset_name: str) -> str:
    """Returns the path to the golden types file for the given dataset."""
    return str(Path(__file__).resolve().parent / "golden" / f"{dataset_name}.types.json")


def get_dataset_object(dataset_name: str) -> DatasetLoader:
    """Returns a Ludwig dataset instance for the given dataset."""
    return ludwig.datasets.get_dataset(dataset_name)
